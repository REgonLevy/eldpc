# Benchmarking eldpc

`bench/bench.py` measures an installed `eldpc` on the machine it runs on and writes two files:
`DIR/results.json` (machine-readable, with the environment of the run) and `DIR/results.md` (the same
numbers as a report).  It needs only what a user of the package has: Python with numpy, scipy and stim,
the installed `eldpc` package (its `libeldpc.so` and its header `eldpc.h`), and a C compiler for the
small C driver.  The built-in bivariate-bicycle and hypergraph-product problems additionally need the
optional qLDPC package (`pip install ".[qldpc]"` from the package root installs the tested revision; see
the README, "Install"); without it the default problem is the surface code, which needs only stim, and
any other code can be benchmarked from a stim detector error model file (`--problem model.dem`).

```bash
python bench/bench.py all --quick --out results        # every measurement, 2-5 minutes (below)
python bench/bench.py all --out results                # the longer default
python bench/bench.py c-latency --quick --out results  # one measurement
python bench/bench.py summary --out results            # the headline numbers of an existing run
python bench/bench.py compare --a results-A --b results-B   # diff two runs; refuses different problems,
                                                            # configs or batch sizes
python -m pytest -q bench/tests                        # the benchmark's own unit tests (no GPU needed)
```

Run it on an otherwise idle machine, with the process pinned to a CPU (`--cpu N`; `--cpu auto`, the
default, picks one: the first CPU of the GPU's NUMA node, from its PCI device's `local_cpulist`, that
is neither CPU 0 nor an SMT sibling of CPU 0, since CPU 0 takes most interrupts and housekeeping work;
else the first allowed CPU that is neither; a warning names the choice and why), and with exclusive
use of the GPU (see "Keeping other GPU users away").

`--device gpu|cpu|auto` chooses the engine (default `gpu`): it adds the front's `device=` key to every
config string, which the C driver and the Python measurements both take, so `--device cpu` measures the
CPU core on a machine without a supported GPU (the README, "Supported GPUs"), and a config string may
carry the CPU core's `threads=` and `cpus=` keys too. A run whose handles ran on the CPU core says so
in its `warnings`, and the core variants of the environment record `cpu`. The CPU core takes several
times the GPU core's time per shot (about 6x on the AVX-512 hosts measured, about 10x on a Zen 2 AVX2
host; the README, "CPU core"), and `--quick` does not scale its shot counts for
it: on the default bb144 xyz problem one `--quick` latency session took about 2 minutes on the CPU core
of a shared 48-core EPYC 7K62 host, so the whole run takes far longer than on a GPU (the RTX 3090 fresh-clone install test of `tools/results/matrix/tested-configurations.json`). Use a small problem there (`--problem
surface`, or your own DEM) or measure one kind at a time (`c-latency`, `ler`). How to pin a CPU-core
run with several threads, and how its sessions differ from GPU sessions, is in "CPU-core runs" below. A GPU create error (no
usable GPU, an unsupported one, a CPU-only build) in a `--device gpu` run names `--device cpu`.  The
results file records the conditions it could check, and a `warnings` list of the ones that were not
met.

`all --quick` on the default problem (bb144, X and Z decoded jointly) takes about 4 to 5 minutes on an
L40S-class GPU with a Xeon host (4 min 32 s in the fresh-clone test of
`tools/results/matrix/tested-configurations.json`; the run of `docs/results/bench-quick.json`, on an
RTX 6000 Ada with an EPYC host, took 216 s; the fresh-clone install test on an RTX 5090 with a Ryzen 7 9700X
desktop host, in the same matrix record, took 2 min 4 s; a loaded host takes longer), most of it the
latency sessions; it prints one
progress line per session, so it can be quiet for a while.  The surface-code problem
(`--problem surface`) is smaller and faster.  On a machine that is not prepared for timing (a CPU governor other than `performance`, other
load, a GPU clock that ramps up during the run) the run ends with a non-empty `warnings` list: that is
expected, and its numbers are screens rather than benchmark results (see "Warnings").

### The C driver

The single-shot and batch C measurements run in `bench/c/bench_driver.c`, which `bench.py` compiles the
first time it needs it (into `DIR/cache/`).  It is an ordinary consumer of the public C API:

- the header is the installed `eldpc.h`.  `bench.py` takes its directory from `ELDPC_INCLUDE`, else from
  `pkg-config --cflags eldpc` (with `<site-packages>/eldpc/lib/pkgconfig` added to `PKG_CONFIG_PATH`),
  else from `<site-packages>/eldpc/include`;
- the library is `libeldpc.so`, taken from `--lib`, else `ELDPC_LIB`, else the path `eldpc.load()`
  reports, else `<site-packages>/eldpc/_libs/libeldpc.so`.  The driver opens it with `dlopen` rather than
  linking it, because one process has to hold two libraries to compare two builds (`--lib` and
  `--lib-b`), and two builds export the same symbol names.  `libeldpc.so` finds its four cores next to
  itself, so no `rpath` is involved.

To build the driver by hand, for example to read it or to run it under a profiler:

```bash
SITE=$(python -c "import eldpc, os; print(os.path.dirname(eldpc.__file__))")
export PKG_CONFIG_PATH=$SITE/lib/pkgconfig
cc -O2 -std=c11 $(pkg-config --cflags eldpc) -o bench_driver bench/c/bench_driver.c -ldl
```

## The four numbers

They measure different things and are never convertible into each other.  Every results file keeps them
apart.

### (a) C API single-shot latency (`c_latency`)

One `eldpc_decode` call per shot, timed with `CLOCK_MONOTONIC` around exactly the call: the syndrome
copy, the kernel launches, the transfers, the synchronisation that makes the outputs valid, and the host
OSD-0 when BP does not converge are all inside.  Creating the decoder, the warm-up shots, reading the
problem and writing results are outside.

Four modes are measured:

- **back to back** -- the next call follows immediately.
- **fixed cadence** -- a busy-wait gap of `--gap-us` (default 1000 us) before every call, with no work
  on the GPU during the gap.
- **fixed cadence with keep-warm** -- the same gap, but `eldpc_keep_warm` is called every 500 us during
  it and once shortly before the call, so the GPU stays at its running clock.
- **fixed cadence, keep-warm against no keep-warm** -- the same decoder measured both ways in the same
  sessions, alternating, so the effect of keep-warm is a paired difference with its own A/A band.

The cadence modes exist because an idle GPU drops its clocks within a few milliseconds: a decoder used
once every millisecond can be much slower than one used back to back, and keep-warm shows how much of
that is recoverable.  A sparse-cadence number without keep-warm is a real number (it is what an
application that decodes rarely will see) but it is not the decoder's latency.

For each variant the results report the pooled per-shot samples over all sessions as nearest-rank p50,
p99, p99.9 and the mean, in microseconds.  Nearest-rank means the reported p99 is an observed shot's
time, not an interpolation.  p99.9 needs thousands of shots to mean anything; the full run collects more
than the quick one for that reason.

**Per-session syndromes.**  Every session of every pair times its own freshly sampled, seeded
syndromes: both handles of a session decode exactly the same shots (the pairing stays exact), and
no two sessions share a shot.  The results record it under `syndromes`:

```
syndromes: {seed, distinct,
            per_session: [{mode, pair, session, shots, seed, sha256}, ...]}
```

`distinct` is the number of distinct sampled shots timed per handle over the sessions of a pair,
the minimum over pairs; each session's `sha256` identifies its shot set, and all of them differ.
A p99 is only as meaningful as the number of distinct shots behind it: **use at least 8000
distinct syndromes per handle per pair for a p99 budget**.  The full defaults give 1500 calls per
session and 6 sessions per pair, i.e. 9000 distinct shots per handle per pair; the keep-warm
pair's two A/A controls also get 6 sessions in that case, so every pair of the full run reaches
9000.  `--shots` and `--processes` scale that number (an explicit value is used as given), and
`--quick` (300 x 4 = 1200) is a screen, not a p99 measurement.

**The slow class.**  Every pooled statistic carries `slow_fraction` -- the share of timed calls
whose status word lacks the converged flag (`ST_BP_CONVERGED`, eldpc.h) -- and `p99_unreliable`,
which is true exactly when `slow_fraction` lies in [0.005, 0.02].  At about 1 % slow calls the 99th
percentile is one of the slow calls and moves with how many happened to fall in the sample, so the
p99 is not a stable number; the results then record the median of the slow calls
(`slow_p50_us`) and the run adds a warning.  Read `slow_fraction` before quoting any p99, and quote
the slow median next to it in that regime.

### (b) Python per-shot latency (`py_latency`)

`Decoder.decode(syndrome)` from Python, the syndrome already a `uint8` numpy array, timed per call with
`time.perf_counter_ns`.  It includes everything the C number includes plus the Python and ctypes
crossing, and the allocation of the returned correction.  It does not include creating the decoder,
warm-up, or converting a syndrome from another dtype.  Compare it with the C number to see what the
Python layer costs; it is the number a Python application sees per call.

Only back-to-back operation is measured here: the Python call is several microseconds of overhead on top
of the decode, so the idle-clock effects of very sparse use are better studied with the C driver, then
added to this number.  Its sessions draw their own seeded syndromes and it records `syndromes`,
`slow_fraction` and `p99_unreliable` exactly as the C measurement does (see "(a)").

### (c) Batch throughput (`throughput`)

`Decoder.decode_batch(syndromes)` over batches of `--batch-shots` (default 1024 quick, 4096 full) shots,
reported as shots per second: total shots divided by total batch time, pooled over sessions.  The batch
size is part of the measurement and is recorded with it.

Two numbers are reported side by side.  `variants` is the Python call: it includes the Python and ctypes
crossing and the allocation and copy of every returned correction and observable vector.  `c_driver` is
the same batch workload through `eldpc_decode_batch` from the C driver with reused output buffers, so it
is the decoder's own batch throughput; `--no-c-batch` skips it.  The difference between the two is the
cost of the Python layer, and it is the number to compare against the decoder's other measurements.

Throughput is not the reciprocal of the single-shot latency.  A batch amortises the per-call overhead
(Python, ctypes, launch setup), but it also materialises every correction and observable vector; the
shots-per-second number can therefore be above or below 1/p50.  Quote it with the batch size it was measured at and never convert between the two.

### (d) Logical error rate (`ler`)

Shots are sampled from the problem's model (the circuit, or the DEM file), decoded with
`decode_batch`, and a shot counts as an error when any predicted observable flip differs from the true
one.  The result is the fraction of wrong shots with a 95% Wilson score interval, which is the right
interval for a binomial count and stays sensible when there are few errors.  A problem with no
observables has no LER and the check reports that instead.

The LER is a property of the problem, the config and the sampler, not of the timing.  Report the number
of shots with it: an LER of 1e-4 measured on 4000 shots is 0 or 1 errors and its interval spans more than
an order of magnitude.  The default run uses 40000 shots.

## Running

```
python bench.py {c-latency,py-latency,throughput,ler,all} [options]
python bench.py summary --out DIR        # the headline numbers of an existing results.json

  --quick                 fewer shots and sessions
  --problem NAME|FILE.dem built-in problem, or a stim detector error model file
  --config NAME[,NAME]    one preset (or ABI config string, or config .json); a second one compares;
                          a preset name may carry extra keys: "a03125c3;threads=8"
  --device gpu|cpu|auto   the engine: the GPU core (default), the CPU core, or the GPU when usable
  --lib PATH              the eldpc C library; --lib-b PATH compares two builds
  --cpu N|auto|none       pin to a CPU (default: auto picks one); 'none' needs --allow-unpinned
  --out DIR               output directory (default: eldpc-bench-results)
  --gap-us G              cadence gap (default 1000)
  --processes N           sessions per pair (default: 4 quick, 6 full; an even count keeps the two
                          session orders balanced)
  --shots N               latency calls per variant per session
  --ler-shots N           shots for the LER check
  --batch-shots N         shots per decode_batch call
  --modes LIST            c-latency modes to run (default: all four)
  --order-design D        'alternating' (default) swaps create and call order together between
                          sessions; 'all4' runs every create/call order combination once per four
  --no-c-batch            skip the C-driver decode_batch half of the throughput measurement
```

Built-in problems (`--problem`): `bb144` (the default when qLDPC is installed: the `[[144,12,12]]`
bivariate bicycle code, 12 rounds, X and Z decoded jointly), `bb144z` (the same code, Z basis), `bb72`,
`bb288`, `hgp98`, and `surface` (rotated surface code, distance 9, 9 rounds; the default when qLDPC is
not installed, and the only built-in that needs nothing but stim).  They build circuit-level memory
experiments with uniform depolarizing noise at p = 1e-3, and derive H, O and the priors from the
circuit's detector error model with the rule of `eldpc.dem_to_arrays` (one column per error instruction
of the flattened DEM, in order, with no columns combined; H and O record the parity of the detectors and
observables it flips; the prior of a column is its probability).  The `surface` built-in derives its DEM
with `decompose_errors=True`; the bivariate-bicycle and hypergraph-product built-ins do not decompose
their errors.  The benchmark's built-ins are fixed: the tuner's `--decompose`/`--no-decompose` options
do not change them, and a problem that must be identical for the tools is passed as one DEM file.

The BB and HGP circuits are built with the qLDPC installed next to `eldpc` (the `qldpc` extra pins the
tested revision, git main commit `4c4f801`).  Different qLDPC
revisions can produce different circuits for the same code and noise (for example, with or without
`REPEAT` blocks, or with different decompositions), so the resulting DEM -- and with it the mechanism
count and the LER -- can differ slightly from a problem built elsewhere.  `results.json` records the DEM
`sha256` and the qLDPC, stim and numpy versions.  To measure exactly the same problem on two machines,
pass the same `model.dem` file to `--problem` on both (every run writes its DEM to `DIR/data/model.dem`);
the built-in names are for convenient like-for-like comparison of machines that use the same package
versions.

A second `--config` (or a second `--lib`) turns on the comparison: the two variants are measured against
each other and the A/A control below is measured next to the difference.  At most two are accepted.

`--cpu` pins the benchmark process, and with it the C driver and the Python workers, to one CPU, using
`sched_setaffinity`.  Timing runs refuse to start unpinned unless `--allow-unpinned` is given, and a run
that could not pin says so in its warnings.

A config may be a preset name followed by extra `key=value` segments, separated by `;`: `--config
"a03125c3;threads=8"` is the preset's string with `threads=8` added, and an extra key that the preset
already has replaces its value (`"a03125c3;max_iter=5"`).  The package and the C API take the same
strings (README, "Presets and choosing one"); `bench.py` expands the name itself, so `results.json`
records the full ABI string, and the label is the string as given.

### CPU-core runs

The CPU core (`--device cpu`, or `auto` without a usable GPU) decodes each shot with `threads=N`
threads (default 1): the calling thread and `N - 1` worker threads, which the handle starts when it is
created and which **busy-wait** between decodes until it is destroyed.  Each worker is pinned to one
CPU: the next entry of the config's `cpus=` list or, without that key, of the CPU affinity of the
thread that creates the handle (repeated when it has fewer CPUs than threads).  Two rules follow.

- **Give every thread its own CPU, and the calling thread the last one.**  Without `cpus=`, `--cpu N`
  pins the whole run to one CPU, which is right for the default `threads=1` and wrong for more threads:
  every worker would land on that one CPU.  For `threads=N`, give the config a `cpus=` list of N CPUs
  of one L3 (one CCD, or one socket) and pin the run with `--cpu` to the list's last entry, the
  calling thread's CPU (the README, "CPU core"); the workers are pinned to the list's first `N - 1`
  entries even though the run itself is pinned to that one CPU:

  ```bash
  python bench/bench.py c-latency --device cpu --config "a03125c3;threads=8;cpus=0-7" --cpu 7 \
      --modes back_to_back --out results-cpu
  ```

  The thread count should equal the size of the `cpus=` list, with no other load on those CPUs and,
  where the machine has SMT, without the SMT siblings of those CPUs in other use.  On the Zen 2 host
  of `tools/results/cpu/zen2-avx2-screen/`, sampling the C driver's threads during such a run
  (`threads=4;cpus=8-11 --cpu 11`) found the three workers on CPUs 8, 9 and 10 and the calling thread
  on CPU 11, each allowed only its own CPU.  The older form `taskset -c 0-7 ... --cpu none
  --allow-unpinned` gives each worker its own CPU too, but leaves the calling thread free to move
  among the CPUs of the set, onto a CPU whose worker is busy-waiting; such a run records
  `pinned_cpu: null` and the "not pinned" warning.  `bench.py` warns when a CPU-core side has more
  threads than distinct CPUs to run them on.
- **Never hold two CPU-core handles at once.**  The paired design below keeps both handles of a
  session alive and alternates their calls.  With the CPU core that would put `2N` spinning threads
  on the `N` pinned CPUs, and the latencies measure the contention rather than the decoder (two
  8-thread handles on 8 CPUs measured milliseconds per bb144 xyz shot instead of the CPU core's
  hundreds of microseconds; `tools/results/cpu/cpu-core-latency-screen.json`).  So when any side of a
  run runs on the CPU core, every session -- C latency, Python latency, both throughput halves --
  times **one handle at a time**: the first handle of the session's call order is created, warmed
  up, timed for all its calls and destroyed before the second is created (the C driver's `--handles
  sequential`, the Python worker's sequential job).  Each handle decodes the session's shots in
  order, so the two handles of a session still decode the same shots and must give the same
  checksum.  The GPU core keeps the interleaved design unchanged.  `results.json` records the design
  as `handle_schedule` (`design: "interleaved"` or `"sequential"`, and the CPU-core sides) in
  `c_latency`, `py_latency`, `throughput` and `throughput.c_driver`.

Why one handle at a time and not half the threads per handle: two handles of `N/2` threads each
would fit the pinned set, but they measure a different configuration (the latency at `N/2` threads
is not the latency at `N`), so the comparison would not be of the config that was asked for.  The
price of the sequential design is the loss of the shot-by-shot alternation: a drift within a session
(frequency, temperature) lands in the difference between the two blocks.  The session order still
alternates which side goes first, so with an even session count (the default) a steady drift cancels
from the pooled difference, and the A/A control measures what is left of it in its band; with the
sequential design the create order is the call order, so `--order-design all4` reduces to the
alternating order.  If an interleaved session reports CPU-core handles anyway (a `device=auto` that
resolved differently in the timed process), the run warns that two CPU-core handles were live at
once.

## The paired A/A design

Every comparison -- config A against config B, build A against build B, keep-warm on against off -- is
measured as follows, and the same machinery run with A on both sides gives the noise band.

- Both sides live in the same process (the C driver for `c_latency`, a Python worker for the
  Python measurements) and are created before any shot is timed.  (CPU-core runs time one handle at
  a time instead: see "CPU-core runs".)
- The calls alternate shot by shot: A, B, A, B, ...
- Several independent sessions (processes) are run for every pair.  Between sessions the order is
  swapped: in half of them A is created and called first, in the other half B is.
- The per-shot samples of all sessions of a side are pooled; the difference of a statistic is computed
  from the pooled sides.
- The A/A control runs A against a second handle of A in exactly the same way.  Its per-session
  differences give the **noise band**: the range of the same statistic that two identical measurements
  produced on this machine under these conditions.  Each comparison carries that band next to its
  difference and a `within_band` flag per statistic.  The control normally runs as many sessions as the
  comparison; the keep-warm pair's two controls (warm against warm, idle against idle) and the C-driver
  throughput half run half as many (at least two), and every results line that quotes a band says how
  many sessions it comes from.
- Throughput bands are built the same way: the band of the B/A shots-per-second ratio is the range of
  that ratio over the A/A control sessions (both sides the same decoder), never over the A/B sessions;
  `within_aa_band` says whether the A/B ratio lies inside it.  The spread of the A/B sessions' own ratios
  is reported separately as `ab_session_ratio_range`.

The A/A band is a range over a handful of sessions, not a confidence interval.  A difference inside the
band is not evidence of a difference; a difference outside it is worth looking at, but with few sessions
the band itself is uncertain, so increase `--processes` and `--shots` before trusting a small one.  Run
a bare `bench.py c-latency` first: its A/A control tells you how well this machine can resolve
differences at all.

The band also contains effects that are not shot noise.  The first handle created in a session can be
measurably faster or slower than the second, and the side called first in each pair of calls can be
faster too.  The alternating design has both orders equal within a session, so its A/A differences
alternate by the sum of the two effects.  The band reports that spread rather than hiding it: it
measures the order effect together with the noise.

Two things follow.  First, the *session count* matters more than the order design: with an odd number
of sessions one order has more sessions than the other, and the pooled A/A difference carries a bias
from the order effect; with an even number the orders are balanced.  The defaults therefore use even
session counts (4 quick, 6 full).  Second, for an odd count the results carry `delta_blocked`, which
averages the per-session differences within each (create, call) order block and then weights the blocks
equally.  `delta_blocked` equals the plain mean when the blocks have equal sizes, and
`within_band_blocked` repeats the band comparison with it.

`--order-design all4` runs every combination of create order and call order once per four sessions.  It
separates the two order effects (which the alternating design confounds) and can lower the standard
error of the per-session difference, but it does not shrink the range band.  The default stays
`alternating`; use `all4` when the create-order and call-order effects need to be told apart, and read
`sem` (the standard error of the mean per-session difference) as the uncertainty of the balanced
estimate.

## Reading `results.json`

```
schema, command, quick, elapsed_s
problem            name, source, m / n / k, recipe, shot seeds and file hashes, DEM sha256
environment        gpu {name, sm_count, l2_bytes, memory_bytes, compute_capability},
                   cpu {model, physical_cores, logical_cores, pinned_cpu,
                        frequency {governor, scaling_driver, energy_performance_preference,
                                   boost ("1", "0" or "unknown"), boost_source, ...},
                        hypervisor, ...},
                   driver, cuda_runtime, toolkit, clocks {before, after}, isa, version, build_info,
                   python_packages, loadavg {before, after},
                   gpu_processes {before, after, between (after each measurement)},
                   core_variants {measurement: {side: "default" | "ada64" | "cpu"}}
warnings           conditions of the run that make a timing unrepresentative
failures           measurements that could not be run, with the error
c_latency          {modes: {back_to_back, cadence, cadence_keep_warm, cadence_keep_warm_pair}:
                       variants {A[, B]}: {pooled {p50_us, p99_us, p99_9_us, mean_us, shots,
                                                   slow_fraction, p99_unreliable, slow_calls,
                                                   slow_p50_us, ...},
                                            status (flag counts and iteration histogram), checksum},
                       order_design, a_a_control: {delta, delta_blocked, per_session_delta,
                                                   order_blocks, band, sem, orders},
                       a_a_controls (keep-warm pair mode): {keep_warm_on, keep_warm_off},
                       comparison: {a_stats, b_stats, delta, delta_blocked, aa_band, within_band,
                                    within_band_blocked} (two variants, or keep-warm against no
                                    keep-warm),
                       keep_warm_stats (keep-warm modes)},
                   handle_schedule {design: "interleaved" | "sequential", what, cpu_core_sides},
                   syndromes {seed, distinct, per_session [...]}}
py_latency         the same shape, one mode (back_to_back)
throughput         {batch_shots, variants {A[, B]}: {shots, seconds, shots_per_s},      (Python)
                    a_a_control {sessions, ratio_b_over_a, per_session_ratio_b_over_a, band_ratio, ...},
                    comparison {ratio_b_over_a, aa_band_ratio, aa_band_sessions, within_aa_band,
                                ab_session_ratio_range, ...},
                    handle_schedule (as in c_latency),
                    c_driver: the same block for eldpc_decode_batch called from C}
ler                {shots, batch_shots, variants {A[, B]}: {errors, ler, ci95 [lo, hi], checksum}}
```

`environment.toolkit` is the CUDA toolkit of the measured library: `{version, nvcc, first_line,
source}`, read from the `cuda=` field of the library's `build_info()` (`source: "build_info"`; the
toolkit that compiled the GPU cores and whose runtime is linked into them, whether or not its
`nvcc` is on `PATH` when the benchmark runs). Only when `build_info()` names no toolkit (a CPU-only
build) does the benchmark ask `nvcc --version` (`CUDACXX`, else the `nvcc` on `PATH`; `source:
"nvcc"`); `null` when neither names one. Results files written before this rule recorded the `nvcc`
on `PATH` only (no `source` key) and show "n/a" when it was not on `PATH`.

The output checksums are hashes of every correction, observable vector and status word of a side.
Every session decodes its own syndromes, so a side's checksum differs between sessions by design.
What must hold is that the two handles of a session -- which decode exactly the same shots --
produce the same checksum; a warning is recorded when they do not, because that would mean the
decoder's outputs are not deterministic for identical inputs (or two handles are not stateless).

## Warnings

The run checks and records the conditions that invalidate timings:

- **another process on the GPU** (nvidia-smi compute applications, sampled before the first measurement,
  after each measurement and after the last; a process that starts and ends inside one measurement is
  not seen, so the record is a lower bound);
- **host load** (`/proc/loadavg`, sampled before and after the run, not during it, so a load peak
  inside the run is not seen; the warning text says so): a warning when the 1-min load average before or
  after the run is above 1 + 5% of the host's logical CPUs: 6.6 on 112 CPUs, 7.4 on 128.  The
  load average counts the whole host's runnable threads, which disturb a timing through the host's
  CPUs, so the threshold does not use the run's pinned CPU mask or a container's cgroup CPU quota
  (`cpu.max`, or `cpu.cfs_quota_us` / `cpu.cfs_period_us`; recorded as `cpu_quota` for
  information).  On a shared host the load measures the other tenants too; a separate warning says
  when the load rose by more than 1 during the run;
- **the CPU governor, frequency and energy-performance preference** of the pinned CPU (a governor or EPP
  other than `performance` can move the clock during a run); the scaling driver, the boost state and the
  SMT sibling of the pinned CPU are recorded with them.  Boost is read from the pinned CPU's
  `cpufreq/boost`, from `/sys/devices/system/cpu/cpufreq/boost` (acpi-cpufreq, amd-pstate) or from
  `intel_pstate/no_turbo`, and is recorded as `unknown` when the machine exposes none of them (virtual
  machines, some drivers);
- **a virtual machine** (the hypervisor CPU flag);
- **GPU clocks before and after** (a large change means the GPU was not at a steady clock);
- **a shot far above the p50** of its side (more than 50 times and above 1 ms): a hiccup the other
  checks can miss;
- **a p99 that sits on the slow-class cliff** (`slow_fraction` in [0.005, 0.02]): the p99 is one of
  the non-converged calls and moves from run to run; the warning names `slow_fraction` and the
  median of the slow calls, which is the stable number in that regime;
- **CUDA runtime and toolkit versions**.

A results file with an empty warnings list was produced under the conditions the benchmark can check.
That is not a guarantee: a machine can be noisy in ways the checks do not see, which is why every
comparison carries the A/A band.

## Comparing two machines

1. Use the same problem.  Pass the same DEM file (`--problem model.dem`) or verify the DEM `sha256` in
   both results files; built-in problems built from different qLDPC revisions can differ.
2. Use the same config string (the results record it) and the same `--quick`/default setting.
3. Compare like with like: `c_latency` p50 with `c_latency` p50, `throughput` with `throughput` at the
   same batch size, `ler` only with overlapping confidence intervals in mind.
4. Quote the A/A band from each machine.  A difference between machines that is smaller than either
   machine's band is not a result.
5. Quote the environment block with the numbers: GPU, SM count, L2, driver, CUDA runtime, toolkit, CPU,
   pinned CPU, governor, clocks, ISA, core variant and package version.

`bench.py compare --a DIR_A --b DIR_B` does steps 1-3 for you: it refuses to print a difference unless
the two runs have the same problem dimensions and DEM hash (or problem file hash), the same config string,
the same batch size and at least one measurement kind in common, and it prints both environments, both
A/A bands and the differences.  It exits 2 with the failed check as its message.

## Usual pitfalls

- **Unpinned runs.**  A process that migrates between CPUs, or shares a core with other work, has a much
  heavier latency tail.  The benchmark refuses to time unpinned unless told to, and warns.
- **An oversubscribed CPU core.**  The CPU core's worker threads busy-wait; more threads than CPUs
  (all threads on the one CPU of `--cpu N` without a `cpus=` list, or a `taskset` set smaller than
  `threads=`) multiplies its latency.  See "CPU-core runs".
- **A busy GPU.**  Time slicing between processes inserts gaps of milliseconds into a latency
  distribution.  The benchmark warns when nvidia-smi shows another compute process; it cannot stop it
  (see the next section).
- **Idle clocks.**  A GPU with no work drops its graphics and memory clocks within milliseconds.  In the
  fixed-cadence mode without keep-warm, the first calls after each gap run at the low clock; keep-warm
  keeps it up.  Compare the cadence modes before concluding that a decoder is slow.
- **The first calls.**  CUDA context creation, kernel loading and the first OSD build happen during
  create and warm-up, never in the measured shots.
- **Too few shots.**  p99.9 from a few hundred shots is the maximum, not a percentile.  LER from a few
  thousand shots has a wide interval; read `ci95`.
- **Batch size.**  Throughput depends strongly on it; never compare two throughput numbers with
  different batch sizes.
- **Different problems.**  LER and latency depend on the code, the number of detectors and mechanisms,
  the syndrome weight and the number of observables.  A number without its problem is not comparable.
- **The Ada core variant at long caps.**  On compute capability 8.9 GPUs with at least 128 SMs the
  package selects the `ada64` core, chosen at the default preset.  With long iteration caps (`ms30`,
  `a05c11`) on bb144 xyz it is slower than the default core (p50 x1.37 at `ms30` on an RTX 6000 Ada,
  x1.285 on an RTX 4090, x1.244 on an L40S with a Sapphire Rapids-W host,
  `tools/results/host-sprw/summary.json`), while the other measured problems stay level or faster;
  see docs/BUILD.md, "Core variants: the Ada 142-SM class".  For long-cap runs on large joint
  problems and for any comparison against a stock-equivalent setting, set
  `ELDPC_DEVICE_CLASS=default` and quote the variant with the numbers (`Decoder.core_variant`, or
  `eldpc.devices()` under `core_variant`).
  The results record the variant of every timed handle (`environment.core_variants`, per
  measurement and side, from `Decoder.core_variant` and, for the C driver, `eldpc_core_variant()`),
  and `results.md` prints it in the environment table.
- **Thermal throttling and frequency boost.**  Long runs heat the GPU and the CPU; the before/after clock
  snapshot and the A/A band help detect drift, but a short run and a long run can differ.

## Keeping other GPU users away

The benchmark does not take a lock on the GPU and does not need one.  To keep a run clean:

- Use a GPU nothing else is using.  `nvidia-smi --query-compute-apps=pid,process_name,used_memory
  --format=csv` lists the compute processes; stop the ones that are not yours, or pick another GPU with
  `CUDA_VISIBLE_DEVICES=N` (the environment record follows that variable, so the report names the GPU
  that was measured).
- On a shared machine, serialise benchmark runs yourself, for example by wrapping the command in
  `flock /path/to/gpu.lock python bench/bench.py all --out results` with a lock file your team agrees
  on, or through the machine's job scheduler.
- Where the administrator allows it, set the GPU's compute mode to exclusive-process
  (`nvidia-smi -c EXCLUSIVE_PROCESS`) so that a second process cannot start on it.  The benchmark itself
  uses several processes in turn (the C driver, the Python workers), never at the same time.
- Do not run a GPU desktop session or another decoder on the same GPU during the run.

`bench.py` records the compute processes it sees (other than itself) before the first measurement,
after each measurement and after the last one, and adds a warning when any appear.  It cannot see a
process that starts and ends inside one measurement.  A run with that warning, or with GPU clock
changes between the start and the end, is not a clean timing; repeat it.

## Troubleshooting

- `cannot find the eldpc C library` -- the package is not installed, or its library is not where the
  benchmark looked (`<site-packages>/eldpc/_libs/libeldpc.so`).  Pass `--lib /path/to/libeldpc.so` (or
  set `ELDPC_LIB`).
- `cannot find eldpc.h` -- the C driver needs the installed header.  Set `ELDPC_INCLUDE` to the
  directory that holds it (`<site-packages>/eldpc/include`) or `PKG_CONFIG_PATH` to
  `<site-packages>/eldpc/lib/pkgconfig`.  The other measurements do not need it; a failed C measurement
  is reported under `failures` and the rest of the run continues with `all`.
- `no C compiler found` -- install `cc`/`gcc`/`clang` or set `CC`.
- `the built-in BB/HGP problems need qLDPC` -- install qLDPC (`pip install ".[qldpc]"` from the package
  root) or pass a DEM file.
  Without qLDPC the default problem is the surface code, and the run records a warning saying so.
- No GPU is visible -- the GPU measurements cannot run; the results record that and warn.

## Files written

`results.json` and `results.md` in `--out`.  Each invocation replaces them, so use `all` (one file with
all four measurements) or a different `--out` per measurement.  The `data/` subdirectory keeps the exact
problem file, the shot files, the DEM, the C driver's per-shot records and the workers' results, so a run
can be inspected and re-analysed without re-running it; the `cache/` subdirectory keeps the compiled C
driver.
