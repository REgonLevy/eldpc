# eldpc tuning knowledge base

This is the strategy and technical base for tuning eldpc for a new code, a new noise level or a new machine. eldpc
runs exact min-sum belief propagation (BP) on the GPU and an exact bit-packed OSD-0 on the host; its decoder core is
the frozen `fastbpgpu6g` (`src/core/`). The document is written for the tuning engineer, or for a coding agent that
runs the tuner on their behalf. It covers what one call does and where its time goes (section 1), every knob with the
semantics the shipped code implements (2), how the knobs interact (3), what has been measured (4), how accuracy is
traded against speed (5), the ideas already measured as losses (6), and how to measure so that a new result is
comparable (7). Known gaps are listed in section 8.

It is a fact book. Every table row names its source; the section "Sources" lists the three kinds (shipped file,
knob-facts key, development record). Shipped files and knob facts are in the package and can be re-checked from it;
development records are not shipped, so a number whose only source is one is named for traceability but cannot be
re-checked from the package: re-measure it (section 7) before relying on it. Numbers are quoted as recorded. "Computed"
marks arithmetic done for this
document on the row's own numbers; it is not a second measurement. Where two records disagree, both are shown with the
condition that separates them.

## 0. Conventions

### 0.1 Exactness classes

| class | meaning | source |
|---|---|---|
| exact | changes no output of any shot (correction, observables, status word, soft output) at the same problem and config | `tests/test_bit_identity.py`; `tests/test_isa.py` |
| lossy | changes accuracy (the logical error rate, LER); a tuning dimension, accepted only by the rule of section 5 | section 5 |
| platform | a deployment or measurement condition (clocks, call cadence, pinning, driver); changes latency, never outputs | section 2.3 |

Every shipped build and every platform option is exact: the package tests compare correction, observables and status shot by shot
against a separately built reference library of the same sources, and the AVX2 and AVX-512 cores against each other
(`tests/test_bit_identity.py`, `tests/test_isa.py`). At the same config the decoder is also bit-identical to the ldpc
reference implementation (development record: parity record, all named reference configs, 0 of 168 compared fields
differ). One consequence matters for every section below: **an LER measured on any machine holds on every machine**
for the same config and shots (cross-machine check: 12M shots, 0 mismatches, `tuning/data/knob_facts.json: gpu_oc`). Only
latency is machine-specific.

### 0.2 Machines

Every latency in this document names one of these machines. Only machine A is a timing-grade setup.

| key | GPU | host | clocks, driver, toolkit | grade | source |
|---|---|---|---|---|---|
| A-oc | RTX 5080 (Blackwell, sm_120, 84 SMs, 64 MiB L2) | Ryzen 7 9850X3D (Zen 5, 8 cores, AVX-512); bench mode: isolated cores, process pinned to one isolated core, performance governor | **GPU and memory overclocked**: +400 MHz graphics offset (3,172-3,187 MHz while decoding vs ~2,880 MHz stock), +3,000 MHz memory offset, power limit 396 W (110 %); driver 615.71.09; library built with CUDA 13.3.73 | timing-grade | `tools/results/final-bench/final-bench.json`; `tuning/data/knob_facts.json: gpu_oc` |
| A-stock | same machine | same | stock clocks, power limit 360 W; driver 615.71.09; CUDA 13.3.73 | timing-grade | `tools/results/final-bench/final-bench.json` |
| B | RTX 5080 (84 SMs) | Threadripper 3970X (Zen 2, AVX2) | clocks not recorded; CUDA 13.3 | development machine: stage shares and ratios | development record: design record |
| C | RTX 5090 (170 SMs) | Ryzen 9 9950X3D (Zen 5), virtual-machine guest | clocks not recorded; CUDA 13.3 | development machine | development record: design record |
| D | RTX 5080 (84 SMs) | Core i9-14900K (AVX2 only) | driver 580.95; CUDA 13.0 / 13.2 / 13.3 | development machine; the registry calls its rows "relative only" | `tuning/data/knob_facts.json: cuda_toolkit`; `tuning/data/knob_facts.json: target_points` (notes) |
| E | RTX 5080 (84 SMs) | Ryzen 9 7950X3D (Zen 4, AVX-512) | driver 595.91.07; CUDA 13.3.73 | **shared development machine, not a timing measurement**; used here for exactness and LER, and for the CPU-core screens of section 4.11 (labelled as screens) | development record: validation runs; development record: CPU-core study; `tools/results/joint-vs-separate/joint-vs-separate.json`; `tools/results/heralded-loss/heralded-loss.json` |
| F | RTX 5090 (170 SMs) | not recorded | driver 575.57.08; CUDA 12.8.93; an older build of the decoder | development machine: probe values of an older build | development record: design record |
| G | RTX 5070 Ti (70 SMs) | EPYC 7532 / 7K62 (Zen 2) | not recorded | development machine (dead-end rows) | `tuning/data/knob_facts.json: order_filter` |
| H | RTX 5080 (84 SMs) | Ryzen 9 9950X (Zen 5) | not recorded | development machine (host-ISA and hand-off rows) | `tuning/data/knob_facts.json: observables_block_scan` |
| I | RTX PRO 4000 | Ryzen 9 7950X3D (Zen 4, V-Cache CCD) | not recorded | development machine (host-ISA rows) | `tuning/data/knob_facts.json: order_filter` |
| J | not recorded | EPYC 7663 | not recorded | shared development machine (dead-end rows) | `tuning/data/knob_facts.json: dead_ends` |
| K | RTX 6000 Ada Generation (AD102, sm_89, 142 SMs, 96 MiB L2, 48 GB, 300 W power limit) | 2x EPYC 7542 (Zen 2, AVX2 only), bare metal (no hypervisor flag) in a container; every timed process pinned to CPU 0 on the GPU's NUMA node | driver 580.126.09; CUDA 13.3.73; SM clock not locked; measurement-harness builds of the final core with the AVX2 host flags | **device-side ratios only**: candidate build against the default build on the same host, pinned core and session; the host is not the deployment class, so no absolute latency or host term from K stands for the deployment machine; another tenant raised the host load during the last long-cap rows (section 4.6) | `tools/results/ada-sm-budget/full-bench.json` (K); `tools/results/ada-sm-budget/rtx6000ada-142sm.json` |
| L | RTX 4090 (AD102, sm_89, 128 SMs, 72 MiB L2, 400 W power limit) | Ryzen 9 7900 (Zen 4, AVX-512), **virtual-machine guest** with other tenants, 16 vCPUs, every timed process pinned to CPU 0 | driver 610.43.02; CUDA 13.3.73; no clock lock; harness builds with the AVX-512 host flags | VM guest: device-side ratios; host timings indicative | `tools/results/ada-sm-budget/full-bench.json` (L); `tools/results/ada-sm-budget/rtx4090-128sm.json` |
| M | L40S (AD102, sm_89, 142 SMs, 96 MiB L2, 350 W power limit, ECC on) | 2x Xeon Silver 4514Y (Emerald Rapids, AVX-512), bare metal in a container (7.68-CPU quota), **host shared with other tenants** (load about 5-7 on 64 CPUs); timed processes pinned to CPU 0, a primary hardware thread on the GPU's NUMA node 0 with its SMT sibling idle | driver 595.84; CUDA 13.3.73; harness builds with the 64-block launch table unless stated | bare metal, other tenants: paired ratios in one process; absolute times indicative | `tools/results/host-xeon/host-xeon.json`; `tools/results/ada-sm-budget/full-bench.json` (`M_earlier_l40s_screen`) |
| N | L40S (AD102, sm_89, 142 SMs, 96 MiB L2, 350 W power limit) | Xeon w9-3495X (Sapphire Rapids-W, 56 cores / 112 threads, one socket and one NUMA node, AVX-512), bare metal (no hypervisor flag) in a container; every timed process pinned to CPU 8 with its SMT sibling idle; `intel_pstate` governor `powersave` with EPP `balance_performance` and turbo on (not changeable from the container) | driver 595.91.07; CUDA 13.3.73; no clock lock; the package built from a clean clone at `21b2820` with its shipped cores | no other GPU process during the runs; bench.py warns about the governor, so not timing-grade: absolute times of this machine; A/B sides as alternating runs | `docs/results/target/`; `tools/results/host-sprw/summary.json` |

**The primary deployment class is measured in halves.** Its GPU half, an RTX 6000 Ada (sm_89, 142 SMs, 96 MiB L2), is
machine K: the device-side effect of the SM budget, the long-cap rows and the create-time shapes were measured on the
real board (section 4.6), on a Zen 2 host with AVX2 only, so they are ratios, not deployment latencies. Its host half,
a Sapphire Rapids Xeon W, is measured on machine N: a Xeon w9-3495X of the same Sapphire Rapids-W family, with an
L40S, the same AD102 chip configuration as the RTX 6000 Ada on a board whose clocks and power limit differ, so N's
absolute times are a proxy for the deployment class, not the board's. On N the shipped package (`21b2820`, AVX-512
core, `ada64` variant, pinned to CPU 8) gives, back to back at `a03125c3`, a C p50 / p99 of 59.03 / 68.57 µs on bb144
xyz (`docs/results/target/bench.json`) and 43.07 / 49.68 µs on bb288 z (`tools/results/host-sprw/summary.json:
bb288_default`); its host-ISA A/B is in section 4.9, and its core-variant A/B (`summary.json: variant_ab`) has the
direction of K and L: `ada64` / default p50 0.544 on bb144 xyz and 0.426 on bb288 z at `a03125c3` (larger gains than
K's 0.740 and 0.649), and 1.244 (slower; K 1.373) on bb144 xyz at `ms30`. Before N, the closest host in the records
was machine M's Emerald Rapids Xeon Silver 4514Y (it supports the same AVX-512 flag set; a two-socket server part,
with an L40S rather than an RTX 6000 Ada), whose host-policy rows are in sections 2.3, 4.4, 4.9 and 4.10
(`tools/results/host-xeon/host-xeon.json`); pinning, SMT
and NUMA placement are measured only there. The package carries SASS for sm_89, and its AVX-512 flag set is one
Sapphire Rapids supports (`CMakeLists.txt`). For scale only: the bb144 xyz `a03125c3` p50 with the 64-SM plan was
64.0 µs on K (AVX2 host core), 61.6 µs on L (both `tools/results/ada-sm-budget/full-bench.json`), 61.7-62.8 µs on M
(the default build's sessions of `tools/results/host-xeon/host-xeon.json`) and 58.7-60.6 µs on N (the shipped package,
11 runs in `tools/results/host-sprw/`), about 1.7-2.0x machine A's 32.12 / 34.02 µs (A-oc / A-stock,
`tools/results/final-bench/final-bench.json`; computed; none of these hosts is timing-grade; section 4.6).

| transfers to another machine | does not transfer |
|---|---|
| LER and every accuracy ratio (outputs are bit-exact) | the absolute latency of any row |
| shot-class fractions (converged vs OSD path) and iteration counts | the per-iteration slope (SM count, clocks, L2, toolkit) |
| the create-time rules of section 2.5, evaluated with the new SM count and L2 size | which shape those rules pick on the new part (verify it, section 2.5) |
| the direction of platform effects (idle clock sag, keep-warm, cadence, launch-ring stalls) | their size (driver, power management, PCIe, host) |
| dead ends that are arithmetic (an inexact change, strictly more work) | dead ends that are platform-scoped (section 6.5) |

### 0.3 Reference problem

`bb144_unif_p1e-3_r12_xyz`: the [[144,12,12]] bivariate-bicycle code, 12 rounds, circuit-level uniform noise at p =
1e-3, X and Z decoded jointly. m = 1,872 detectors, n = 76,032 mechanisms, nnz(H) = 438,912, k = 24 observables,
rank(H) = 1,860 (development record: design record; `tuning/data/knob_facts.json: observables_block_scan`). These are
the sizes of the development corpus, whose circuits were built with the released qLDPC 0.4.1; the package builds the
problem with qLDPC revision `4c4f801`, which gives the same m and k but n = 75,600 and nnz(H) = 437,184
(`docs/results/bench-quick.json`; `docs/VERIFY.md`, "Problems"; the verify suite's circuit builder run under qLDPC
0.4.1 reproduces 76,032 and 438,912). Other problems in the records: section 3.3. A new code enters through H (m x n,
CSC), O (k x n, CSC) and n priors;
`Decoder.from_dem` builds them from a stim detector error model with one column per `error` instruction of
`dem.flattened()`, in order, never combined or dropped (`python/eldpc/_dem.py`).

### 0.4 The tuning workflow

| step | what to do | what decides it | section |
|---|---|---|---|
| 1 | describe the problem: m, n, nnz(H), max row degree, k, nnz(O), basis (z-type or joint xyz), noise points | max row degree and 8 x nnz(H) choose the formulation; n and nnz set the iteration and OSD costs; LER surfaces differ by code and by p; whether joint or separate X/Z decoding is more accurate depends on the preset | 2.5, 3.3, 3.5 |
| 2 | measure stock's LER at every noise point you must cover (ldpc common settings; eldpc `ms30` is bit-identical to it) | acceptance is relative to stock at the same p | 5.1 |
| 3 | start from configs by code family: joint xyz: `a03125c3`, (0.375, 5), `a0375c7`; z-type: `a04375c3` at p = 1e-3, alpha 0.5 with caps >= 15 at higher p | the measured surfaces | 4.2, 4.3 |
| 4 | measure LER paired on fresh shots at two or more noise points: shots, failures, one-sided bound | the bound must clear the bar; the cap's cost grows with p | 4.3, 7 |
| 5 | on the deployment machine: cap sweep at fixed alpha, fit `t_fixed + cap x t_iter`, judge bimodal presets per class; verify the create-time shape; on an Ada part with the `ada64` core, time both core variants at the chosen cap | the slope does not transfer between machines; on Ada the faster variant depends on the cap and the problem | 4.1, 2.5, 4.6 |
| 6 | build the Pareto table (latency against LER per noise point) and take the fastest acceptable configuration | section 5 rules | 5 |
| 7 | set the platform: call cadence and keep-warm, pinning, GPU clocks, host ISA, toolkit, core variant | measured platform effects | 2.3, 4.4-4.10 |
| 8 | report with the checklist | comparability | 7.3 |

### 0.5 Build labels and the two-point slope

| label | meaning | source |
|---|---|---|
| 6g, "final build" | the shipped build: the frozen core `fastbpgpu6g` in `src/core/` | `src/core/fastbpgpu6g.cpp` |
| 6f, 6e, 6d, 6c | earlier builds of the same decoder line, newest first. The registry predates 6g: its `components` fit and most of its platform rows are 6e on machine A, its cap rows are 6c, and 6f has no machine-A row | `tuning/data/knob_facts.json: components`; `tuning/data/knob_facts.json: target_points` |
| legacy base | `fastbpgpu3_a`, the exact base before builds 5 and 6, ~2.7x slower per iteration than 6e | `tuning/data/knob_facts.json: components_legacy` |
| older build | a development build before 6g whose label the record does not give (machine F probes, many dead-end rows) | development record: design record |

The per-iteration estimator used for build-to-build comparisons is the **two-point slope**
`(T(cap 31) - T(cap 1)) / 30`, measured in one session (section 7.1). A "two-point slope ratio" is the slope with a
change divided by the slope of the build it changes.

---

## 1. What one call does and where its time goes

### 1.1 The call sequence (`decoder=bposd`, one shot)

| step | what happens | cost (machine) | source |
|---|---|---|---|
| 1 | argument checks; an all-zero syndrome returns zeros, `BP_CONVERGED` and 0 iterations without a GPU launch | entry 0.02 µs, prep 0.04 µs p50 (B) | `src/core/fastbpgpu6g.cpp` (file header); development record: design record |
| 2 | syndrome (m bytes) to the device through pinned staging | host side 0.02 µs p50 (B); the H2D of m = 252-2,736 bytes was 4.13-4.54 µs on an older build (F) | development record: design record |
| 3 | one cooperative kernel: first check pass, the BP iterations (check pass + column pass, two grid barriers each), trailing mismatch pass; on a non-converged shot the device also selects the OSD-0 candidate prefix (rung one, section 1.4); the status word is published into mapped pinned memory | device body 32.74 µs p50 at cap 3 (build 6e, A-oc) | `src/core/bp_gpu.cu`; `tuning/data/knob_facts.json: components` (device_split_cap3) |
| 4 | the host polls the mapped status (bounded by an iteration budget; no clock read on the decode path) | launch + observe ~3.3 µs p50 / ~12.4 µs p99, including a ~9 µs launch-ring stall on ~6-7 % of back-to-back calls (build 6e, A-oc, driver 615) | `tuning/data/knob_facts.json: components` (device_split_cap3) |
| 5 | converged shot: n decision bytes D2H; no posterior copy | 4.66-5.88 µs for n = 2,664-76,032 on an older build (F) | development record: design record |
| 6 | non-converged shot: the host takes the candidate records, walks the OSD-0 order (GF(2) elimination, back-substitution); a walk past rung one triggers rung two on the device; past that, the full posterior (exact fallback) | elimination 4.31 / 1.53 µs p50, back-substitution 0.90 / 0.57 µs p50 (B / C) | `src/core/osd_caps.hpp`; development record: design record |
| 7 | observables: XOR of O's columns over the correction's support | 0.14 µs p50 (B and C) | development record: design record |
| 8 | status word written (section 1.2) | – | `include/eldpc.h` |

Host part of the whole call at cap 3: 4.36 µs p50 / 13.75 µs p99 (call minus launch-to-observed-status; build 6e,
A-oc; `tuning/data/knob_facts.json: components`). Three facts of the call path that affect tuning:

- `eldpc_decode_batch` runs the single-shot path once per shot (`src/core/fastbpgpu6g.cpp`, `dbench_decode_batch`). It
  saves per-call API overhead, not GPU time; every shot's result equals `eldpc_decode` on it (`include/eldpc.h`).
- The soft output is copied back lazily: `eldpc_get_soft_output` after a converged shot triggers an 8n-byte D2H. Keep
  it out of the latency path (`src/core/fastbpgpu6g.cpp`, `dbench_get_soft_output`).
- The device-side "mapped tail" that hands rung two over through mapped memory runs only at `max_iter <= 3`
  (`kOsdTailMaxIter = 3`, `src/core/bp_gpu.cu`); at longer caps it delayed the next launch more than it saved, so caps
  above 3 take the stream-ordered copy.

### 1.2 The status word

`out_status = (flags & 0xFFFF) | (iterations << 16)`, iterations saturating at `0xFFFF`. Flags: `BP_CONVERGED` 0x1
(no post-processing ran), `POSTPROC_INVOKED` 0x2 (OSD ran), `HIGHER_ORDER_WIN` 0x4 (only with `osd_order > 0`),
`ABSTAIN` 0x8, `SYNDROME_UNMATCHED` 0x10 (always set on a non-converged `decoder=bp` shot) (`include/eldpc.h`;
`src/core/fastbpgpu6g.cpp`, file header). Per-class analysis (section 3.2) splits on `BP_CONVERGED`; the iteration
count shows how far below the cap the converged shots stop.

### 1.3 Where the time goes

**The measured line on the final build** (machine A, alpha 0.5, caps 3-31; section 4.1 has the rows, from the
`tools/results/final-bench/final-bench.json`): p50 = 13.83 +
6.193 x cap µs (A-oc) and 14.16 + 6.734 x cap µs (A-stock), max residual 0.13 / 0.08 µs; p99 = 26.58 + 6.220 x cap µs
(A-oc) and 26.78 + 6.776 x cap µs (A-stock) (computed, least squares over the nine rows of section 4.1). The intercept
holds the launch, the copies, the fixed device passes and the OSD path at its median; the p99 intercept is ~12.7 µs
above the p50 one, which is the OSD tail.

**Stage breakdown of the final build** (two development machines; shares transfer better than µs):

| stage | B (RTX 5080 + 3970X): p50 / p99 / mean µs | C (RTX 5090 + 9950X3D): p50 / p99 / mean µs | source |
|---|---|---|---|
| whole call (stage-probe build) | 42.59 / 56.24 / 43.26 | 33.23 / 46.32 / 34.90 | development record: design record |
| BP engine call (device) | 36.81 / 38.69 / 36.86 | 30.25 / 40.45 / 31.85 | development record: design record |
| OSD-0 elimination (host) | 4.31 / 17.42 / 5.02 | 1.53 / 11.04 / 2.01 | development record: design record |
| OSD-0 back-substitution (host) | 0.90 / 1.26 / 0.92 | 0.57 / 0.97 / 0.60 | development record: design record |
| observables (host) | 0.14 / 0.49 / 0.14 | 0.14 / 0.35 / 0.17 | development record: design record |
| entry, prep, syndrome copy, hand-off, order init, exit, status, return | each <= 0.07 p50 / <= 0.22 p99 | same | development record: design record |
| elimination share of the call | 10.1 % p50, 31.0 % p99 (computed) | 4.6 % p50, 23.8 % p99 (computed) | computed from the rows above |

Reading: the median call is mostly the device; the host OSD walk is a tail term and grows on the slower host (Zen 2 vs
Zen 5). Both runs are at a03125c3 (cap 3), where essentially every bb144 xyz shot takes the OSD path (section 3.2). A decision
taken on one host must be re-checked on the other.

**Fixed costs measured by probes** (older build of the decoder, machine F; model inputs, not shot latencies):

| term | value | source |
|---|---|---|
| empty kernel launch + stream synchronisation | 5.35 µs (p10 5.07, p90 5.67) | development record: design record |
| flag round trip through mapped pinned memory to a polling kernel (1 block) | 3.24 µs (32 polling blocks: 98.0 µs) | development record: design record |
| posterior D2H (8n bytes, pinned), n = 2,664 / 10,512 / 31,392 / 76,032 | 4.52 / 5.58 / 8.42 / 14.67 µs | development record: design record |
| `cg::this_grid().sync()`, two per iteration | bb144 z 1.30, bb288 z 1.58, bb144 xyz 2.40 µs per iteration | development record: design record |
| idle grid barrier, 16-128 blocks x 128; `__syncthreads()`; `cluster.sync()` (4 or 8) | 569-590 ns; 17.7-18.0 ns; 143 ns | development record: design record |

**Earlier decompositions that must not be mixed with the line above.** The registry's fit of build 6e on A-oc is p50 =
16.07 + 8.191 x cap, p99 = 27.08 + 8.481 x cap; it is derived (one measured point at cap 3: 40.6 / 52.3 µs, the other
caps from build 6c rows plus per-class deltas) and folds the OSD path into the intercept
(`tuning/data/knob_facts.json: components`). The legacy base (fastbpgpu3_a) had t_iter 22.0 µs, t_fixed 28.7 µs and a separate OSD path of 59.5 / 110
µs p50 / p99, measured on A at stock clocks (`tuning/data/knob_facts.json: components_legacy`). Different builds, different
decompositions; do not add a term of one to the other.

### 1.4 The OSD path

OSD-0 runs only on shots where BP did not converge. The device hands the host a prefix of the reference OSD-0 column
order, in rungs:

| element | shipped value / rule | measured | source |
|---|---|---|---|
| rung one (appended to the BP kernel) | the first `kOsdPrefixKA = 384` entries of the order, or the longest prefix that fits `kOsdPrefixCapA = 640` | bb144 xyz: the rung-one host walk exhausts on 3 of 2,000 shots | `src/core/osd_caps.hpp`; `tuning/data/knob_facts.json: osd_cap_rungs` |
| rung two (standalone repick kernel) | `kOsdPrefixK = 1024`, cap `kOsdPrefixCap = 2048` (= the host buffer `kCand`, asserted) | repick rate 0 at every preset in one measurement | `src/core/osd_caps.hpp`; `src/core/osd0_engine.hpp`; `tuning/data/knob_facts.json: osd_cap_rungs` |
| full-posterior fallback | a walk past rung two copies the 8n-byte posterior and walks it on the host | exact by construction | `src/core/osd_caps.hpp` |
| device sort of the handed-over records | only when n >= `kOsdSortMinN = 16384` (bb144 z, n = 10,512, keeps the host's cheap walk) | bb144 xyz host stage of 3.64 µs p50 / 10.11 µs p99 removed at a05ms31 (an RTX 5080 development machine, host not recorded) | `src/core/bp_gpu.cu` (comment) |
| threshold sample and histogram | stride sample `kOsdS = 512` keys; histogram `kOsdHBins = 256` bins over [-2.5 P, +2.5 P], P = largest finite prior LLR (6.91 on bb144 xyz), fixed when the priors are set | rung-one set p99 at a03125c3: 528 / 457 / 431 records at 128 / 256 / 512 bins; window +-1.5 P..+-6 P flat | `src/core/bp_gpu.cu`; `tuning/data/knob_facts.json: osd_histogram_bins` |
| kept rank buckets `osd_r` | 8 x the estimated threshold bucket (`kOsdPrefixK x kOsdS / n`), floor `kOsdRMin = 64`, capped at the sample size | recorded values 64 (bb144 xyz), 128 (bb288 z), 392 (bb144 z), 512 (bb72 z); too small gives full-order fallbacks, too large wastes bucket work | `src/core/bp_gpu.cu`; `tuning/data/knob_facts.json: osd_r_truncated_rank` |

OSD cost on three builds (different definitions): the final build's elimination + back-substitution is 5.21 µs p50 on
B and 2.10 µs on C (computed sums of section 1.3); build 6e's p99 OSD term is 11.0 µs (p99 minus p50 intercept, A-oc,
`tuning/data/knob_facts.json: components`); an older build's OSD-0 term (bposd minus bp, non-converged class) was 8.3 /
16.6 / 38.7 / 75.9 µs on bb72 z / bb144 z / bb288 z / bb144 xyz, including the posterior D2H the final build no longer
does (F; development record: design record).

### 1.5 Whole-call numbers and where they disagree

All rows: bb144 xyz at a03125c3 (alpha 0.3125, cap 3), back-to-back calls.

| build / condition | p50 / p99 (µs) | p99.9 | source |
|---|---|---|---|
| final build, A-oc, single handle, 4 processes | 32.12 / 42.37 | 63.39 | `tools/results/final-bench/final-bench.json` |
| final build, A-stock, single handle, 4 processes | 34.02 / 44.15 | 66.08 | `tools/results/final-bench/final-bench.json` |
| final build, B, solo, 12 + 12 alternating processes of 2,000 shots | 33.69 / 48.65 | 90.29 | development record: design record |
| final build, B, paired, 3 + 3 processes per create order, 12,000 shots per side | 35.3 / 50.7 | 109.78 | development record: design record |
| final build, B, stage-probe build (instrumented), 4,000 shots | 42.59 / 56.24 | – | development record: design record |
| final build, C, AVX-512 core, 24,000 solo shots | 32.92 / 46.82 | 108.91 | development record: design record |
| build 6e, A-oc, registry row | 40.6 / 52.3 | 77.3 | `tuning/data/knob_facts.json: target_points` |
| build 6e, A-oc / A-stock, final-bench session | 40.30 / 52.24; 42.61 / 54.68 | 73.64; 76.31 | `tools/results/final-bench/final-bench.json` |
| final minus 6e, paired, pooled | -8.08 / -9.45 (A-oc); -8.33 / -9.95 (A-stock) | – | `tools/results/final-bench/final-bench.json` |
| build 6c, A-oc | 44.0 / 56.0 | 77.2 | `tuning/data/knob_facts.json: target_points` |
| legacy base (fastbpgpu3_a), A-stock / A-oc | 156.6 / 205.3; 152.8 / 201.5 | – | `tuning/data/knob_facts.json: bench` (legacy_vs_5early_stock_oc) |

Machine B's solo row lies between A-oc and A-stock (+1.6 / -0.3 µs), its paired row is 3.2 µs above A-oc, and the
stage-probe row 10.5 µs above A-oc because of the instrumentation (computed); B's clocks are not recorded. The
registry's 6e row and the final bench's 6e row on the same machine agree within 0.3 µs. Quote a latency with its
machine, clocks and build.

---

## 2. The knobs and what each one does

### 2.1 Run-time config keys

The config is an ABI string `key=value;key=value` (or, in Python, a preset name, a dict or a `.json` path). The parser
is `build_config` in `src/core/fastbpgpu6g.cpp`; it reads a map, so key order does not matter, and it follows the ldpc
reference's key, whitespace and duplicate rules.

| key | required | accepted values | default | what it does | class | source |
|---|---|---|---|---|---|---|
| `decoder` | yes | `bposd`, `bp` (`bplsd` rejected) | – | `bposd`: BP, then the configured OSD on every non-converged shot. `bp`: BP only; a non-converged shot returns the hard decision with `SYNDROME_UNMATCHED`; any `osd_*` / `lsd_*` key is rejected | lossy (bp vs bposd) | `src/core/fastbpgpu6g.cpp` (`build_config`) |
| `bp_method` | yes | `ms` only (`ps` rejected) | – | min-sum BP | – | same |
| `max_iter` | yes | integer >= 1 | – | the BP iteration cap; a shot that converges earlier exits early, a shot at the cap goes to OSD | lossy (section 4.2) | same |
| `ms_scaling_factor` (alpha) | no | float > 0 | 0.625 | min-sum scaling of the check-to-bit messages; time-neutral (< ~1 µs) | lossy (section 4.2) | same; `tuning/data/knob_facts.json: components` (fit_note) |
| `schedule` | no | `parallel` only (`serial` rejected) | `parallel` | flooding schedule | – | same |
| `osd_method` | **yes for `bposd`** | `osd0`, `osd_e`, `osd_cs` | – | `osd0`: the library's own exact bit-packed OSD-0 (shots whose syndrome is outside the image of H take the vendored path); `osd_e` / `osd_cs`: the vendored ldpc exhaustive / combination-sweep OSD | lossy vs `osd0` | same |
| `osd_order` | no | integer >= 0; must be 0 for `osd0`, <= 15 for `osd_e` | 0 | order of the higher-order search | lossy vs 0 | same |
| `lsd_method`, `lsd_order` | – | recognised keys, rejected for both decoders | – | – | – | same |
| any other key | – | rejected ("unknown config key") | – | – | – | same |

Python front (`python/eldpc/_config.py`): `resolve_config` takes a preset name, a dict, an ABI string or a `.json`
path; ABI string segments must contain exactly one `=`, keys match `[a-z0-9_]+`, duplicates are an error.
`config_string(dict)` gives the canonical string (keys sorted). The priors are part of the problem, not of the config:
`Decoder(H, O, priors=...)` or `error_rate=` for uniform priors; `Decoder.set_priors(p)` (C: `eldpc_set_priors` with
LLRs ln((1-p)/p)) replaces them for later decodes without rebuilding the handle (`python/eldpc/__init__.py`;
`include/eldpc.h`). The OSD histogram window is
re-derived from the priors when they are set (`src/core/bp_gpu.cu`).

The vendored `osd_e` / `osd_cs` paths are bit-identical to the ldpc reference at the same method and order, and
roughly 20-45x slower per decode than OSD-0 on the large problems (development record: parity record).

### 2.2 Presets

Exactly the table of `python/eldpc/_config.py` (`_PRESET_PARAMS`). Every preset is `decoder=bposd`, `bp_method=ms`,
`schedule=parallel`, `osd_method=osd0`, `osd_order=0`; every preset also has a `.bp` variant (same alpha and cap,
`decoder=bp`, no OSD keys). **The default is `a03125c3`**: it is the default of `Decoder(...)` and
`Decoder.from_dem(...)` (`python/eldpc/__init__.py`); `_config.py` itself marks no default, and the C API has none
(pass a string).

| preset | `ms_scaling_factor` | `max_iter` | note | source |
|---|---|---|---|---|
| `ms30` | 0.625 | 30 | ldpc's common settings; bit-identical to stock, so its LER is stock's | `python/eldpc/_config.py` |
| `ms100` | 0.625 | 100 | no tuning record | `python/eldpc/_config.py` |
| `ms30a075` | 0.75 | 30 | alpha outside the confirmed 0.25-0.5 window; no tuning record | `python/eldpc/_config.py`; `tuning/data/knob_facts.json: alpha` |
| `a05ms31` | 0.5 | 31 | the reference of the tuning records (bb144 xyz p = 1e-3 LER 1.28e-5) | `python/eldpc/_config.py`; `tuning/data/knob_facts.json: ler_reference` |
| `a05c15` | 0.5 | 15 | bb144 z at p = 2e-3 needs alpha 0.5 and caps >= 15 | `python/eldpc/_config.py`; `tuning/data/knob_facts.json: alpha` |
| `a05c11` | 0.5 | 11 | within noise of the reference at p = 1e-3 and 2e-3 (one of three 2e-3 screens worse); 1.027x at 3e-3 | `python/eldpc/_config.py`; section 4.3 |
| `a0375c7` | 0.375 | 7 | the largest LER margin among the presets on bb144 xyz at p = 2e-3 and 3e-3 | `python/eldpc/_config.py`; section 4.3 |
| `a04375c3` | 0.4375 | 3 | the tuned bb144 z setting at p = 1e-3 (section 5.5) | `python/eldpc/_config.py` |
| `a03125c3` | 0.3125 | 3 | **default**; fastest acceptable on bb144 xyz at p <= 2e-3; worse than the reference at 3e-3 | `python/eldpc/_config.py`; `python/eldpc/__init__.py`; section 5 |

Canonical ABI string of a preset (`config_string`, keys sorted), shown for `a03125c3`; the others differ only in
`max_iter` and `ms_scaling_factor` (formatted as Python `repr`, e.g. `0.625`, `0.4375`):
`bp_method=ms;decoder=bposd;max_iter=3;ms_scaling_factor=0.3125;osd_method=osd0;osd_order=0;schedule=parallel`; its
`.bp` variant is `bp_method=ms;decoder=bp;max_iter=3;ms_scaling_factor=0.3125;schedule=parallel`
(`python/eldpc/_config.py`).

Configs that are not presets but sit on the measured frontier (section 5.3): (0.375, 5), (0.3125, 7), (0.375, 9). Pass
them as strings.

### 2.3 Platform and deployment knobs

| knob | setting | default | what it does | class | source |
|---|---|---|---|---|---|
| keep-warm | `eldpc_keep_warm(h, ELDPC_KEEP_WARM_GPU)`, `Decoder.keep_warm()`; core: `fastbp_keep_warm(h, FASTBP_KEEP_WARM_GPU)`; flag value 1 is the only one defined | not called; the library starts no thread | enqueues one empty 1-block kernel on the handle's stream: ~1.5 µs host time, no sync, allocation, thread or I/O; keeps the GPU clock up during idle gaps. Use it only at a call cadence: back to back it gains nothing at p50 and can cost the next call's p99 (M: bb288 z p99 x1.127; section 4.4) | exact (bit-identical; section 4.4) | `src/core/fastbp_ext.h`; `include/eldpc.h`; `python/eldpc/__init__.py` |
| call cadence | gap between decode calls | – | at a 10 ms gap without keep-warm, +4.7 µs p50 at A-oc and +2.5 µs at A-stock over back-to-back; flat to 100 µs. On the Ada boards with the 64-block plan (bb144 xyz, separate sessions, indicative): +3.5 to +5.3 µs on M (two session pairs), +8.9 µs on K (section 4.4) | platform; state it with every number | `tools/results/final-bench/final-bench.json`; `tuning/data/knob_facts.json: call_cadence`; `tools/results/host-xeon/host-xeon.json`; `tools/results/ada-sm-budget/full-bench.json` (K) |
| thread placement | one handle per thread (distinct handles may run concurrently); pin the decoding thread to an isolated core, on a multi-socket host to a core on the GPU's NUMA node whose SMT sibling is idle (either hardware thread of that core measures the same); `taskset -c <cpu>` is enough. On a dual-CCD V-Cache part, the V-Cache CCD (the records' Zen 4 V-Cache rows were pinned there; no CCD A/B of the GPU core is recorded) | unpinned; the library starts no thread | removes migration; the host OSD walk is cache-sensitive. On M a busy SMT sibling costs +3.3 to +6.4 % p50 and +7.4 to +9.1 % p99, unpinned +0.7 to +11.1 % p50, the other socket +0.5 to +2.3 % p50 (section 4.10); print the pinned CPU with every host timing | platform | `include/eldpc.h`; `tuning/data/knob_facts.json: order_filter`; `tuning/data/knob_facts.json: dead_ends`; `tools/results/host-xeon/host-xeon.json` |
| core variant | `ELDPC_DEVICE_CLASS=default` or `ada64` in the environment of the creating process (read at every create; empty = unset; any other value fails the create) | the device-class table: compute capability 8.9 with >= 128 SMs gets `ada64` (the core built with `DBENCH_GPU_SM_BUDGET=64`), every other device `default` | `ada64` plans grids and launches for 64 SMs: faster at `a03125c3` on the 142- and 128-SM Ada parts, slower on bb144 xyz at `a05c11` and `ms30` (section 4.6); set `default` for long-cap presets on large joint problems and for comparisons at stock-equivalent settings, or time both | exact (bit-identical outputs; `Decoder.core_variant` reports a handle's variant) | `src/port/core_variant.h`; `python/eldpc/_variant.py`; `docs/BUILD.md` ("Core variants") |
| GPU selection | the core always uses device 0 of the visible devices: choose with `CUDA_VISIBLE_DEVICES`; `eldpc.devices()` lists SM count and L2 | device 0 | – | platform | `src/core/bp_gpu.cu` (`setup`); `python/eldpc/_devices.py` |
| GPU clocks (overclock) | validated profile on A: +400 MHz graphics, +3,000 MHz memory, 110 % power | stock | final build -4.0 to -7.7 % (section 4.5) | platform; outputs bit-identical (12M cross-machine shots, 0 mismatches; 0 Xid over 13.3M OC shots) | `tuning/data/knob_facts.json: gpu_oc`; `tools/results/final-bench/final-bench.json` |
| locked graphics clocks (`nvidia-smi -lgc`) | pin the clock at the boost | not set | expected to remove most of the idle clock sag without keep-warm launches | platform; **untested, no number** | `tuning/data/knob_facts.json: locked_gfx_clocks` |
| IOMMU passthrough | `amd_iommu=pt` / `iommu=off` for the GPU group | translated (DMA-FQ) | estimate only: p99 about -2.2 µs, mean about -0.4 µs (the launch-ring stall share) | platform; **untested** | `tuning/data/knob_facts.json: iommu_passthrough` |
| host writes of the shot block through a BAR1 mapping | GDRCopy-style module | – | estimate only: p50 about -0.35 µs, p99 about -2.0 µs, from an RTX PRO 4500 measurement | **not shipped**; untested on any deployment machine | `tuning/data/knob_facts.json: gdrdrv_host_writes` |
| host ISA | `ELDPC_ISA=avx2` or `avx512` (read once at load), or `eldpc.load(isa=...)` before the first load | CPUID/XGETBV choice: AVX-512 core when the CPU and OS support AVX-512F/VL/DQ/BW | selects the host build of the OSD walk and helpers; forcing `avx512` on a CPU without it is a load error, not a fallback. On an AVX-512 host keep the automatic choice: on M the AVX2 core is about 4-13 % slower on bb144 xyz and bb288 z at `a03125c3`, and `-march` builds gain nothing (section 4.9) | exact (both cores bit-identical) | `src/port/eldpc_front.cpp`; `python/eldpc/_abi.py`; `tests/test_isa.py` |
| API layer | Python `Decoder.decode` vs the C call | – | Python adds per-call overhead; every latency here is the C call. Measure the difference with `tools/measure_decode_overhead.py` | platform | `tools/measure_decode_overhead.py` |
| batch memory | Python `decode_batch` always allocates corrections: shots x n bytes (75,600 B per shot on the package's bb144 xyz = 75.6 GB at 1M shots, computed) | – | decode LER sets in chunks, or through the C API with `out_corrections = NULL` | – | `python/eldpc/_abi.py`; `include/eldpc.h` |

**Keep-warm, exactly as shipped.** `src/core/fastbp_ext.h` defines one flag, `FASTBP_KEEP_WARM_GPU 1u`
(`include/eldpc.h`: `ELDPC_KEEP_WARM_GPU 1u`; Python `eldpc.KEEP_WARM_GPU = 1`). `flags = 0` does nothing. The call
returns OK; `EINVAL` for a NULL handle **or any flag bit other than 1**; `EINTERNAL` when the launch failed (the
handle stays usable; the next decode reports any sticky device error). The Python method raises on a non-zero return.
Contract: call it every <= 0.5 ms while the handle is idle (sparser calls let the clocks drop); the last call is best
placed shortly before the next decode; never concurrently with another call on the same handle. A host-table sweep
flag (value 2) existed in an unshipped variant and is still described in `tuning/data/knob_facts.json: keep_warm_host_sweep`;
the shipped core rejects it with `EINVAL`, and it was removed because a well-placed sweep gained nothing and a sweep
alone evicted more than it warmed (`src/core/fastbpgpu6g.cpp`, comment above `fastbp_keep_warm`; section 6.5).

### 2.4 Build-time knobs

| knob | shipped value | what it does | class | source |
|---|---|---|---|---|
| CUDA architectures | SASS for sm_89 and sm_120, PTX for compute_120 | an architecture outside these (and below compute 12.0) needs the list extended, then the exactness tests re-run | exactness re-checked per build | `CMakeLists.txt` |
| CUDA toolkit | 13.3 (nvcc 13.3.73) for every row of section 4 | code generation: on 13.0 the record kernels spill (REG 128, STACK 40 B, 68 STL/LDL sites), on 13.3 they have an 8 B frame (section 4.8); record kernels sit at the 128-register cap | exactness re-checked per build | `tuning/data/knob_facts.json: cuda_toolkit` |
| nvcc exactness flags | `-fmad=false -ftz=false -prec-div=true -prec-sqrt=true`, last on every nvcc line | required for bit identity | exact | `CMakeLists.txt` |
| host flags | `-O2 -DNDEBUG -ffp-contract=off -fno-fast-math`, then `-O3`, then the ISA flags | required for bit identity | exact | `CMakeLists.txt` |
| ISA flag sets | AVX2 core `-mavx2`; AVX-512 core `-mavx2 -mavx512f -mavx512vl -mavx512dq -mavx512bw` ("never a wider set") | two cores from the same sources | exact | `CMakeLists.txt` |
| `DBENCH_GPU_SM_BUDGET` | 0 = use the device's SM count, in the default cores; 64 in the two `ada64` cores (the only definition that differs; `ELDPC_CORE_DEFINITIONS` in `CMakeLists.txt`) | caps `sms = min(device SMs, budget)`; `sms` drives the grid caps, the formulation rule and the launch shape; emulates a smaller part on a larger one, and on the Ada 142-SM class is the shipped launch plan (section 4.6) | exact (budget builds bit-identical, section 4.6) | `CMakeLists.txt`; `src/core/bp_gpu.cu` |
| `DBENCH_GPU_L2_BUDGET` | 0 = use the device's L2 size | caps `l2 = min(device L2, budget)` in the formulation rule | exact | same |
| `DBENCH_LAYOUT_BALANCE` | 1 (macro default in the source) | degree-aware placement of 32-column groups on a block's warps; 0 restores the older degree-sorted order for probe builds | exact by construction (a create-time permutation of whole groups) | `src/core/bp_gpu.cu` |
| test-only macros | `FASTBP_OSD_PREFIX_{KA,CAPA,K,CAP}` (rung sizes for forced-fallback tests), `FASTBP_TEST_*`, `DBENCH_TEST_LATE_INIT`, `DBENCH_STAGE_PROBES`, `BP_FP_DECISION` (FP64 form of the decision test, value-equal), `DBENCH_INEXACT_MEASURE` (0) | never set in a shipped build | not for shipping | `src/core/osd_caps.hpp`; `src/core/bp_gpu.cu` |

The `build_info` string (`eldpc.build_info()`) records compiler, flags, CUDA version, architectures, ISA and the
budgets. Its `gpu_sm_budget=...; gpu_l2_budget=...` tail is generated from the same definition list the core is
compiled with (`CMakeLists.txt`), so the two cannot drift apart: the `ada64` cores' own strings carry
`gpu_sm_budget=64` (`eldpc.build_info()` reports the default core of the loaded ISA; `docs/BUILD.md`), and a budget
probe build that changes `ELDPC_CORE_DEFINITIONS` reports its own value.

### 2.5 What the library chooses at create (and is still a knob)

The create-time shape is a function of H, the SM count and the L2 size. A new code or a new part changes it, and a
shape that fails a gate silently takes the slower path, so verify it before timing.

| choice | shipped rule | measured points | source |
|---|---|---|---|
| warps per row `wpr` | 1 if max row degree <= 64, else `min(8, ceil(max row degree / 64))` | bb72 / bb144 / bb288 z have `wpr = 1`; bb144 xyz does not | `src/core/bp_gpu.cu` (`setup`) |
| formulation (in-place vs row-summary) | summary when `too_wide` (wpr > 1 and the in-place grid request `ceil(m x wpr / 8)` exceeds `per_sm x sms`) **or** `8 x nnz(H) > l2` **or** `wide_grid` (the in-place grid needs more blocks than the part has SMs). A shape chosen by `wide_grid` alone is kept only if it ends on the fused summary kernel with the b2c window; otherwise create starts over without that term (one retry) | older builds (source comment and F): bb144 xyz takes the summary (645.8 vs 951.9 µs); the in-place path wins where it fits (bb72 loss problem 350.4 vs 535.4 µs; that problem was built with the development corpus's herald insertion, which section 5.6 corrects, so its H is not the verify suite's); summary +18 % on bb144 z, +7 % on bb288 z, -32 % on bb144 xyz | `src/core/bp_gpu.cu` (`setup`, the wide-grid retry); development record: design record |
| interleaved 32-column edge-slot layout (ILV) vs CSC order | ILV only when its residual waste `(steps - sum(deg)) / sum(deg)` <= 12 % (`kIlvWasteLimitPct`) | older build, G (reproduced on H: -20.4 % end to end on bb144 xyz): bb144 xyz 1.9 % waste (1.44x faster per iteration); bb288 z 18.9 % (-10 % shot time); bb144 z 23.6 % (+6 %) and hgp98 z 47.1 % (+10 % per iteration) keep CSC | `src/core/bp_gpu.cu` (comment); `tuning/data/knob_facts.json: ilv_layout` |
| block balance across blocks | the balanced column partition replaces the uniform one only if it predicts a strictly lower slowest block (`kBalMarginPct = 0`) and its row footprint keeps the side of the 255-row gate and grows by at most a quarter (`ns_bal x 4 <= ns_uni x 5 + 8`) | predicted slowest-block gain at 84 blocks: bb144 xyz 8.92 %, bb144 z 0 %, bb72 z 0 %, hgp98 xyz -3.29 %; bb288 z not balanced at 84 SMs; refused for bb144 xyz at 70 blocks | `src/core/bp_gpu.cu`; `tuning/data/knob_facts.json: balance_gate` |
| shared-memory b2c window | needs the SM's opt-in shared memory (101,376 B on the RTX 5080); falls back to the same kernel with global b2c, then to the non-row-slot kernel | bb144 xyz at 84 blocks: 9 rounds, 73,728 B window, 95,216 B request: fits; at 170 SMs 57,584 B; a 70-SM part falls back and loses the gain (cap 3 -1.7 µs; two-point slope ratio 0.866 -> ~0.92); z presets have no window | `tuning/data/knob_facts.json: b2c_window` (machine D) |
| OSD rungs, sort gate, `osd_r`, histogram | section 1.4 | section 1.4 | `src/core/osd_caps.hpp`; `src/core/bp_gpu.cu` |
| mapped tail | only at `max_iter <= 3` | section 1.1 | `src/core/bp_gpu.cu` |

How to see the shape: the shipped library does not expose it. A test build of the core with
`FASTBP_TEST_LAYOUT_PROBE=1` exports `fastbp_test_layout(h, out, n)`, which returns the create-time `LayoutStats`
words (sms, nblocks, nthreads, sum_mode, ilv, balanced, rslot, ns_max, the balance gate, window rounds and bytes,
opt-in limits) (`src/core/fastbpgpu6g.cpp`; `src/core/bp_engine.hpp`). The eldpc front does not forward it; call it on
the core library of that build.

Most measured gains of the late builds are on the bb144 xyz row-slot kernels; the z presets were flat to -4 µs per
class; fallback kernels that no recorded problem takes grow their stack on 13.3, so time a new problem that lands on
them with that in view (`tuning/data/knob_facts.json: kernel_shape_choice`).

---

## 3. Interactions

### 3.1 alpha with max_iter

| interaction | measured effect | source |
|---|---|---|
| alpha is time-neutral, the cap sets the time: choose alpha on LER first, then the cap on latency | alpha moves the line by < ~1 µs (final build on A-oc, cap 3: 32.12 at alpha 0.3125 vs 32.54 at 0.5; cap 7: 56.84 at 0.375 vs 57.20 at 0.5); each cap step costs the slope of section 4.1 | `tools/results/final-bench/final-bench.json`; `tuning/data/knob_facts.json: components` |
| even caps at alpha 0.5 | BP oscillates with period 2: (0.5, 30) LER 8.5e-5 at 800k shots vs 1.28e-5 at cap 31 (6.6x); on the legacy base even caps also had a worse p99 than the next odd cap (A-oc). Use odd caps | `tuning/data/knob_facts.json: cap`; `tuning/data/knob_facts.json: alpha` (notes); `tuning/data/knob_facts.json: bench` (low_caps_oc) |
| caps 1 and 2 | cap 1 fails everywhere; cap 2 is acceptable only at alpha 0.3125 (3M-shot confirmation) | `tuning/data/knob_facts.json: alpha` |
| the low-alpha optimum is code- and noise-specific | bb144 xyz: alpha 0.3125-0.375 best at every p; bb144 z at p = 2e-3: alpha 0.5 with caps >= 15, every other alpha 1.2-3.3x worse | `tuning/data/knob_facts.json: alpha` |
| surface tested | alpha 0.3125-0.75 x odd caps 3-31 on bb144 xyz at p = 1e-3 (2M shots), 2e-3 (500k), 3e-3 (300k), and bb144 z at 2e-3 (1M) (alpha x cap surface screen) | `tuning/data/knob_facts.json: alpha` |

### 3.2 The convergence fraction and the OSD share

The OSD path runs only on non-converged shots, so the converged fraction decides how much of the distribution the OSD
walk shapes. Fractions are properties of config and shots, not of the machine.

| problem / config | shots on the OSD path | source |
|---|---|---|
| bb144 xyz, a03125c3 (0.3125, 3) | 99.97 % (319,904 of 320,000, A); 100 % (12,000 of 12,000, E) | `tools/results/final-bench/final-bench.json`; development record: validation runs |
| bb144 xyz, a0375c7 | 97.9 % (A); 97.4 % (E, 11,688 of 12,000) | same two |
| bb144 xyz, a05c11 | 98.1 % (A); 97.55 % (E) | same two |
| bb144 xyz, a05ms31 | 95.1 % (A); 94.3 % (E); 94.7 % (older build; registry `frac_converged` 0.053 is marked stale) | same two; `tuning/data/knob_facts.json: components` |
| bb144 z, a04375c3 / a05c15 / a05ms31 | 60.9 / 46.1 / 44.3 % | `tools/results/final-bench/final-bench.json` |
| bb288 z, a05ms31 | 83.9 % | `tools/results/final-bench/final-bench.json` |

Consequences. On bb144 xyz at low caps the OSD path is the whole distribution after BP: the median is BP plus the
walk, and the p99 is set by the walk (section 1.3: elimination is 10 % of the p50 call and 31 % of the p99 on B). On
the z problems at alpha 0.5 the distribution is bimodal: bb144 z a05ms31 on A-oc has a converged class at 54.31 µs p50
and an OSD class at 124.39 µs p50, so the pooled p50 (61.60 µs) describes neither; judge such presets per class
(`tools/results/final-bench/final-bench.json`; `tuning/data/knob_facts.json: kernel_shape_choice`; `tuning/data/knob_facts.json: dead_ends`,
"mapped converged decisions judged on the pooled p50 of a bimodal preset"). At cap 3 on bb144 xyz the few converged
shots are slower than the OSD shots (36.02 vs 32.12 µs p50, A-oc): the converged path takes the n-byte decision copy
instead of the mapped hand-over (`tuning/data/knob_facts.json: mapped_decisions`).

### 3.3 Code size

Sizes of the development corpus (section 0.3: its bivariate-bicycle circuits are the ones qLDPC 0.4.1 builds). The
package's qLDPC revision builds 2,592 / 10,368 / 31,104 / 75,600 / 2,352 columns for these five names
(`docs/VERIFY.md`, "Problems").

| problem | m x n | nnz(H) | k | per iteration, older build (F) | fixed part at `max_iter = 1`, older build (F) | source |
|---|---|---|---|---|---|---|
| `bb72_unif_p1e-3_r6_z` | 252 x 2,664 | 9,072 | 12 | 4.66 µs | 19.41 µs | development record: design record |
| `bb144_unif_p1e-3_r12_z` | 936 x 10,512 | 35,856 | 12 | 4.78 µs | 21.36 µs | same |
| `bb288_unif_p1e-3_r18_z` | 2,736 x 31,392 | 107,136 | 12 | 6.34 µs | 24.46 µs | same |
| `bb144_unif_p1e-3_r12_xyz` | 1,872 x 76,032 | 438,912 | 24 | 14.39 µs | 42.90 µs | same |
| `hgp98_unif_p1e-3_r4_z` | n = 2,450 | – | 18 | – | 19.67 µs | same |

- The per-iteration term is far from linear in n: 0.513, 0.133, 0.059 and 0.033 ns per edge-iteration for bb72 / bb144
  / bb288 / bb144 xyz (older build, F). Small problems are launch- and barrier-bound; the grid barrier alone is
  1.30-2.40 µs per iteration (section 1.3).
- The fixed part varies by session on the same build: other sessions gave 19.33 / 22.85 / 24.02 / 42.90 µs and
  19.2 / 20.0 / 25.0 / 43.4 µs for bb72 z / bb144 z / bb288 z / bb144 xyz (development record: design record).
- The fixed part grows much more slowly than n (19.4-42.9 µs over a 28x range of n): launch and copies dominate it
  (same source).
- The observables term scales with nnz(O): on the older build the column-XOR form took 0.446 µs on bb72 z (nnz(O)
  6,570) and 12.79 µs on bb144 xyz (nnz(O) 289,944); the final build's block scan is 0.14 µs p50 on bb144 xyz (B, C)
  (development record: design record).
- A code with uniform column degrees, or with block chunks too small for homogeneous warp groups, fails the 12 % ILV
  gate and keeps CSC on any machine (section 2.5).

### 3.4 Noise level

The cap's cost and the alpha optimum both move with p (tables in sections 4.2-4.3, frontier in 5.4):

- bb144 xyz, alpha 0.5: at p = 1e-3 the cap is flat from 5 to 31 within ~+-35 % (cap 7: 1.0 [0.712, 1.4]; cap 11:
  0.886-0.935 over 3M-10M shots); at 2e-3 cap 7 is 1.153 [1.05, 1.26] (500k-shot cap screen) and cap 11 lost in one
  screen and not in the next two (1.114, 1.049, 0.986); at 3e-3 cap 11 is 1.027 [1.001, 1.054]
  (`tuning/data/knob_facts.json: cap`).
- bb144 xyz, the default a03125c3: 0.591 / 0.85-0.923 / 1.066-1.074 of the reference at p = 1e-3 / 2e-3 / 3e-3,
  while a0375c7 stays at 0.62-0.74 at every p (`tuning/data/knob_facts.json: alpha`; development record: validation runs).
- bb144 z: a04375c3 is 10x below stock at p = 1e-3, but at 2e-3 low alpha at cap 7 is 1.5-3.5x worse than the
  reference (`tools/results/stock-baseline/stock-baseline.json`; `tuning/data/knob_facts.json: alpha`).

One noise point is not enough to choose a preset; the registry's rule is to quote the cap's cost per noise point
(`tuning/data/knob_facts.json: cap`, notes).

### 3.5 Joint (xyz) against separate X and Z decoding

A circuit with both X and Z detectors can be decoded jointly (one handle on the full DEM, the bb144 xyz problems) or
as two sector problems, X and Z, each decoded on its own handle. The separate construction here projects the joint
DEM onto each sector's detector rows (936 X-type and 936 Z-type detectors on bb144 xyz), drops the columns with no
detector in that sector (none of them flips a sector observable), and merges identical columns exactly (independent
mechanisms combined by parity, p = p1 (1 - p2) + p2 (1 - p1)): 8,784 columns and 30,672 nonzeros per sector, against
76,032 joint columns. Without the merge the sector projection is worse (below). Both constructions decode the same
shots; a shot fails if any of its 24 observables is wrong; LER per shot of the 12-round experiment. Accuracy only, no timing (the separate route makes two
decode calls per shot). Machine E, harness build of the final core; `tools/results/joint-vs-separate/joint-vs-separate.json`.

| problem | config | shots | joint LER (failures) | separate LER (failures) | separate / joint | joint-only / separate-only | exact McNemar p | more accurate |
|---|---|---|---|---|---|---|---|---|
| bb144 xyz p = 1e-3 | `a03125c3` | 5,000,000 | 1.060e-5 (53) | 1.720e-5 (86) | 1.623 | 34 / 67 | 1.3e-3 | joint |
| same, fresh-seed confirmation | `a03125c3` | 1,000,000 | 1.0e-5 (10) | 1.0e-5 (10) | 1.0 | 8 / 8 | 1.0 | tie |
| bb144 xyz p = 1e-3 | (0.3125, 31) | 3,000,000 | 9.33e-6 (28) | 8.33e-6 (25) | 0.893 | 23 / 20 | 0.76 | tie |
| bb144 xyz p = 1e-3 | (0.5, 3) | 3,000,000 | 1.667e-5 (50) | 8.67e-6 (26) | 0.520 | 42 / 18 | 2.7e-3 | separate |
| bb144 xyz p = 1e-3 | `a05ms31` | 2,000,000 | 1.600e-5 (32) | 3.5e-6 (7) | 0.219 | 31 / 6 | 4.1e-5 | separate |
| same, fresh-seed confirmation | `a05ms31` | 1,000,000 | 1.6e-5 (16) | 5.0e-6 (5) | 0.31 (computed) | 16 / 5 | 0.027 | separate |
| bb144 xyz p = 3e-3 | `a03125c3` | 200,000 | 9.310e-3 | 1.730e-2 | 1.858 | 915 / 2,513 | 3.1e-170 | joint |
| bb144 xyz p = 3e-3 | `a05ms31` | 200,000 | 8.865e-3 | 3.135e-3 | 0.354 | 1,538 / 392 | 6.5e-160 | separate |
| hgp98 xyz p = 1e-3, 4 rounds | `a03125c3` | 1,000,000 | 1.413e-3 | 5.731e-3 | 4.056 | 365 / 4,683 | log10 p = -951.9 | joint |
| hgp98 xyz p = 1e-3, 4 rounds | `a05ms31` | 1,000,000 | 1.634e-3 | 5.439e-3 | 3.329 | 705 / 4,510 | log10 p = -674.1 | joint |

Other rows of the same record: with `decoder=bp` (no OSD-0; 200,000 shots, bb144 xyz p = 1e-3) separate is better at
both presets (separate / joint 0.971 at `a03125c3`, 0.266 at `a05ms31`); the sector projection without the exact
column merge is worse than with it on the same 500,000 fresh shots (2.4e-5 merged vs 5.2e-5 unmerged at `a03125c3`,
McNemar p 2.6e-3; 0 vs 2.6e-5 at `a05ms31`, p 2.4e-4). A control decoded the joint model with its 4,752 duplicate
columns merged exactly (71,280 columns; `Decoder.from_dem` never merges): 41 failures against the joint model's 53 on
the same 5M shots at `a03125c3`, 35 against 32 on the 2M at `a05ms31`; the record has no paired test of merged against
unmerged joint, so treat it as descriptive. The main runs used a stopping rule fixed before the first
shot (stop when the two 99 % Clopper-Pearson intervals are disjoint at a 1M-shot checkpoint): `a05ms31` stopped at
2M, `a03125c3` ran to 5M.

Readings. (1) The direction depends on the preset, so never state a preset-free "joint is better": at the shipped
`a03125c3` joint decoding was the more accurate on bb144 xyz at p = 1e-3 (1.6x in the 5M-shot run; the 1M-shot
fresh-seed confirmation tied, 10 vs 10; pooled 63 vs 96 failures in 6M shots, computed) and at 3e-3 (1.9x); at the
alpha 0.5 presets separate decoding won (4.6x at (0.5, 31), 1.9x at (0.5, 3), 2.8x at p = 3e-3). (2) The record reads
the crossed caps as the scaling factor driving the flip: at cap 3, alpha 0.3125 -> 0.5 turns 1.623 into 0.520, while at
alpha 0.3125 raising the cap to 31 only removes the joint advantage (0.893, tie); the numbers are as recorded, the
causal reading is the record's. (3) On hgp98 xyz joint decoding wins at both presets (3.3-4.1x). (4) For a new circuit,
decode both constructions on the same shots at the deployment preset and noise point; this is a lossy choice, accepted
by the rule of section 5, and the separate route's latency (two calls, or two handles in parallel) must be measured.

---

## 4. Measured effects

### 4.1 Latency against max_iter

Final build, bb144 xyz, alpha 0.5, single handle, back-to-back, 3 processes per row (4 for caps 11 and 31); machine A
at both clock profiles (`tools/results/final-bench/final-bench.json`).

| cap | A-oc p50 / p99 (µs) | A-stock p50 / p99 (µs) | OSD-path share | registry build 6e line, p50 (computed from its fit) | build 6c on A-oc, p50 / p99 |
|---|---|---|---|---|---|
| 3 | 32.54 / 46.38 | 34.44 / 48.17 | 99.7 % | 40.64 | 44.0 / 56.0 (alpha 0.3125) |
| 5 | 44.74 / 57.11 | 47.75 / 60.15 | 99.7 % | 57.03 | 62.6 / 73.9 (alpha 0.3125) |
| 7 | 57.20 / 69.95 | 61.32 / 74.06 | 98.8 % | 73.41 | 81.9 / 94.1 (alpha 0.375) |
| 9 | 69.54 / 82.46 | 74.78 / 87.61 | 98.6 % | 89.79 | – |
| 11 | 81.94 / 95.11 | 88.27 / 101.26 | 98.1 % | 106.17 | 120.4 / 135.5 |
| 13 | 94.27 / 106.93 | 101.71 / 114.68 | 97.2 % | 122.55 | – |
| 15 | 106.71 / 119.58 | 115.13 / 128.25 | 96.8 % | 138.94 | – |
| 21 | 143.89 / 157.27 | 155.50 / 168.89 | 96.1 % | 188.08 | – |
| 31 | 205.87 / 219.71 | 222.98 / 237.22 | 95.1 % | 269.99 | 309.0 / 330.1 |

Sources of the last two columns: `tuning/data/knob_facts.json: components` (fit_note) and
`tuning/data/knob_facts.json: target_points` (6c rows, A-oc).

| per-iteration estimator | value | machine / build | source |
|---|---|---|---|
| slope of the measured p50 line, caps 3-31 | 6.193 µs (A-oc), 6.734 µs (A-stock) | A, final build | computed from the table above |
| slope of the registry's derived line | 8.191 µs p50, 8.481 µs p99 | A-oc, build 6e | `tuning/data/knob_facts.json: components` |
| two-point slope (T(cap 31) - T(cap 1)) / 30 in one session, 3 + 3 processes per create order | 7.449 µs | B, final build | development record: design record |
| two-point slope with the grid barrier stripped (probe build, not exact) | 6.562 µs: the barrier is 0.890 µs of the iteration | B | development record: design record |
| legacy base, (cap 31 - cap 1) / 30 | 22.0 µs stock, 21.0 µs OC | A, fastbpgpu3_a | `tuning/data/knob_facts.json: components_legacy` |

Do not use one estimator to correct another: they differ in build, machine, clocks and method. The registry's "~2 x
t_iter per odd step (~38-46 µs)" matches the legacy base's t_iter (2 x 22.0 = 44 µs, computed;
`tuning/data/knob_facts.json: cap`, per_step_us); on the final build
an odd step (cap + 2) costs ~12.4 µs p50 on A-oc and ~13.5 µs on A-stock (computed: 2 x slope).

Per class at A-oc (`tools/results/final-bench/final-bench.json`): a0375c7 converged 51.21 / OSD 56.87 µs p50; a05c11 65.68 / 81.97;
a05ms31 102.46 / 206.05. The converged class exits early, so a preset whose converged share grows with the cap gets a
sub-linear p50 at long caps.

### 4.2 LER against alpha and max_iter, bb144 xyz at p = 1e-3

"Ratio" is the paired failure ratio against (0.5, 31) on the same shots, with its 95 % interval; LER is failures /
shots (computed). LERs hold on every machine (section 0.1).

| alpha | cap | shots | failures | LER | ratio vs (0.5, 31) | stage | source |
|---|---|---|---|---|---|---|---|
| 0.625 | 30 | 800,000 | 174 | 2.2e-4 | stock; the reference is 0.058 [0.05, 0.067] of it (~17x lower) | – | `tuning/data/knob_facts.json: ler_reference`; `tuning/data/knob_facts.json: alpha` |
| 0.5 | 31 | 12,500,000 | 160 | 1.28e-5 | reference | pooled | `tuning/data/knob_facts.json: ler_reference` |
| 0.5 | 31 | 5,000,000 | 68 | 1.36e-5 | reference (5M-shot cap sweep) | – | `tuning/data/knob_facts.json: ler_reference` |
| 0.3125 | 3 | 17,300,000 | 160 | 9.3e-6 (99 %: 7.5e-6 to 1.13e-5) | – | five fresh-shot runs on the identical circuit (13 / 2M, 35 / 3M, 41 / 4.3M, 26 / 4M, 45 / 4M), consistent with one rate (chi-square p 0.07) | `tools/results/stock-baseline/stock-baseline.json` |
| 0.3125 | 3 | 3,000,000 | – | 1.17e-5, one-sided CP99 1.71e-5 | – | 3M-shot confirmation | `tuning/data/knob_facts.json: alpha` |
| 0.3125 | 3 | 2,000,000 | – | – | 0.591 | alpha x cap surface screen | `tuning/data/knob_facts.json: alpha` |
| 0.3125 | 7 / 11 / 31 | 2,000,000 | 19 / 21 / 24 | 9.5e-6 / 1.05e-5 / 1.2e-5 | 0.864 [0.503, 1.5] / 0.955 [0.559, 1.64] / 1.091 [0.644, 1.83] (ref 22) | 2M-shot alpha x cap screen | `tuning/data/knob_facts.json: alpha` |
| 0.375 | 7 / 11 / 31 | 2,000,000 | 12 / 17 / 21 | 6.0e-6 / 8.5e-6 / 1.05e-5 | 0.545 [0.314, 1.02] / 0.773 [0.454, 1.35] / 0.955 [0.576, 1.59] (ref 22) | 2M-shot alpha x cap screen | `tuning/data/knob_facts.json: alpha` |
| 0.4375 | 7 / 11 / 31 | 2,000,000 | 12 / 16 / 22 | 6.0e-6 / 8.0e-6 / 1.1e-5 | 0.545 [0.321, 0.996] / 0.727 [0.45, 1.21] / 1.0 [0.642, 1.56] (ref 22) | 2M-shot alpha x cap screen | `tuning/data/knob_facts.json: alpha` |
| 0.375 | 7 | 10,000,000 | 92 | 9.2e-6 | 0.742 [0.59, 0.938], McNemar p 0.0146 (ref 124); vs a05c11 0.793 [0.634, 0.997] | 10M-shot confirmation, pre-registered | `tuning/data/knob_facts.json: alpha` |
| 0.4375 | 7 | 10,000,000 | 100 | 1.0e-5 | 0.806 [0.671, 0.973], p 0.0293 (ref 124) | 10M-shot confirmation | `tuning/data/knob_facts.json: alpha` |
| 0.5 | 11 | 10,000,000 | 116 | 1.16e-5 | 0.935 [0.813, 1.08], p 0.416 (ref 124) | 10M-shot confirmation | `tuning/data/knob_facts.json: cap` |
| 0.5 | 3 / 5 / 7 / 9 / 11 | 3,000,000 | 46 / 33 / 35 / 36 / 31 | 1.53e-5 / 1.1e-5 / 1.17e-5 / 1.2e-5 / 1.03e-5 | 1.314 [0.897, 1.9] / 0.943 [0.654, 1.36] / 1.0 [0.712, 1.4] / 1.029 [0.751, 1.41] / 0.886 [0.674, 1.17] (ref 35) | 3M-shot cap screen | `tuning/data/knob_facts.json: cap` |
| 0.5 | 9 / 11 / 13 / 15 / 21 | 5,000,000 | – | – | 0.882 [0.701, 1.11] / 0.882 [0.729, 1.07] / 0.882 [0.732, 1.07] / 0.971 [0.819, 1.15] / 1.088 [0.949, 1.25] | 5M-shot cap sweep | `tuning/data/knob_facts.json: cap` |
| 0.5 | 30 | 800,000 | – | 8.5e-5 | 6.6 (even cap) | – | `tuning/data/knob_facts.json: cap` |

The 3M-shot confirmation (3M shots at p = 1e-3 plus 500k at 2e-3): every cap >= 3 is acceptable at alpha 0.25-0.5
(CP99 < 1e-4, no cliff); cap 1 fails everywhere; cap 2 passes only at 0.3125 (`tuning/data/knob_facts.json: alpha`).
Two records give the a03125c3 LER: 1.17e-5 on 3M shots (3M-shot confirmation) and 9.3e-6 on 17.3M pooled shots that
include those 3M; the five runs are consistent with one rate.

### 4.3 LER at other noise levels and on bb144 z

**bb144 xyz at p = 2e-3** (paired; the reference count differs by screen):

| alpha | cap | shots | failures | ratio [95 % CI] | McNemar p | ref failures | screen | source |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 7 / 11 | 500,000 | – | 1.153 [1.05, 1.26] / 1.114 [1.03, 1.2] | 0.00207 / 0.00567 | 334 | 500k-shot cap screen | `tuning/data/knob_facts.json: cap` |
| 0.375 | 7 | 500,000 | 185 | 0.647 [0.559, 0.752] | 7.6e-9 | 286 | 500k-shot alpha x cap screen | `tuning/data/knob_facts.json: alpha` |
| 0.4375 | 7 / 11 | 500,000 | 217 / 223 | 0.759 [0.666, 0.867] / 0.78 [0.69, 0.883] | 4.9e-5 / 9.6e-5 | 286 | 500k-shot alpha x cap screen | `tuning/data/knob_facts.json: alpha` |
| 0.5 | 11 | 500,000 | – | 1.049 [0.961, 1.14] | 0.322 | 286 | 500k-shot alpha x cap screen | `tuning/data/knob_facts.json: cap` |
| 0.375 | 5 / 7 / 9 | 500,000 | 225 / 222 / 218 | 0.63 [0.554, 0.719] / 0.622 [0.548, 0.709] / 0.611 [0.536, 0.698] | 2e-12 / 3e-13 / 1e-13 | 357 | three-point alpha x cap screen | `tuning/data/knob_facts.json: alpha` |
| 0.3125 / 0.4375 | 7 | 500,000 | 236 / 271 | 0.661 [0.579, 0.757] / 0.759 [0.682, 0.846] | 1.1e-9 / 5.5e-7 | 357 | three-point alpha x cap screen | `tuning/data/knob_facts.json: alpha` |
| 0.5 | 11 | 500,000 | 352 | 0.986 [0.911, 1.067] | 0.78 | 357 | three-point alpha x cap screen | `tuning/data/knob_facts.json: cap` |
| 0.3125 / 0.375 / 0.3125 | 7 / 5 / 3 | 500,000 | – | 0.649 / 0.728 / 0.887 | – | – | alpha x cap surface screen | `tuning/data/knob_facts.json: alpha` |
| 0.3125 | 3 | 500,000 | – | 0.85 | – | – | 3M-shot confirmation | `tuning/data/knob_facts.json: alpha` |
| 0.3125 | 3 | 1,000,000 | 622 | 0.923, one-sided 95 % bootstrap upper 0.990 | 0.0809 (401 vs 453 discordant) | 674 | re-noised variant | development record: validation runs |

**bb144 xyz at p = 3e-3** (three-point alpha x cap screen: 300,000 paired shots, reference 2,620 failures,
`tuning/data/knob_facts.json: alpha`; the re-noised variant: 500,000 shots, reference 4,427 failures, development
record: validation runs):

| alpha | cap | failures | LER (computed) | ratio [95 % CI] | McNemar p | screen | source |
|---|---|---|---|---|---|---|---|
| 0.375 | 5 / 7 / 9 | 2,082 / 1,897 / 1,859 | 6.94e-3 / 6.32e-3 / 6.20e-3 | 0.795 [0.763, 0.828] / 0.724 [0.694, 0.756] / 0.71 [0.679, 0.741] | 1.5e-28 / 5.1e-51 / 3.0e-55 | three-point alpha x cap screen | `tuning/data/knob_facts.json: alpha` |
| 0.3125 / 0.4375 | 7 | 1,853 / 2,228 | 6.18e-3 / 7.43e-3 | 0.707 [0.674, 0.742] / 0.85 [0.821, 0.881] | 8.3e-47 / 5.2e-19 | three-point alpha x cap screen | `tuning/data/knob_facts.json: alpha` |
| 0.5 | 11 | 2,691 | 8.97e-3 | 1.027 [1.001, 1.054] | 0.0427 | three-point alpha x cap screen | `tuning/data/knob_facts.json: cap` |
| 0.3125-0.375 | >= 7 / 5 | – | – | 0.67-0.73 / 0.79-0.82 | – | alpha x cap surface screen | `tuning/data/knob_facts.json: alpha` |
| 0.3125 | 3 | – | – | 1.074 | – | alpha x cap surface screen | `tuning/data/knob_facts.json: alpha` |
| 0.3125 | 3 | 4,721 | 9.442e-3 | 1.066, one-sided 95 % bootstrap upper 1.094: **fails** | 6.1e-5 (2,821 vs 2,527) | re-noised variant | development record: validation runs |
| 0.375 | 7 | 3,224 | 6.448e-3 | 0.7283, upper 0.7497; a03125c3 / a0375c7 on the same shots 1.4643 (upper 1.5006) | 2.4e-79; 4.1e-143 | re-noised variant | development record: validation runs |

The re-noised variants keep the code, rounds and basis and change `noise.p` to 0.002 or 0.003; each is a new
fresh-seed shot set (1,000,000 and 500,000 shots, 0 abstains), decoded on machine E
(LER does not depend on the machine; development record: validation runs). Their absolute LERs (reference 6.74e-4 and
8.854e-3) are not comparable with the
screens' (different shots); only the paired ratios are. Three records for a03125c3 at p = 2e-3 (0.85, 0.887, 0.923)
are compatible: the variant's paired test is not significant. At p = 3e-3 the variant agrees with the surface screen
(1.066 vs 1.074): the default preset's advantage at p <= 2e-3 does not carry to 3e-3, while cap 7 at alpha 0.375 stays
~27 % below the reference.

**bb144 z at p = 2e-3** (three-point alpha x cap screen: 1,000,000 paired shots, reference 121 failures;
`tuning/data/knob_facts.json: alpha`, `tuning/data/knob_facts.json: cap`):

| alpha | cap | failures | ratio [95 % CI] | McNemar p |
|---|---|---|---|---|
| 0.375 | 5 / 7 / 9 | 295 / 241 / 242 | 2.438 [2.059, 2.859] / 1.992 [1.689, 2.33] / 2.0 [1.698, 2.336] | 3e-30 / 2e-18 / 5e-19 |
| 0.3125 | 7 | 425 | 3.512 [2.946, 4.139] | 3.5e-60 |
| 0.4375 | 7 | 182 | 1.504 [1.292, 1.742] | 7.3e-8 |
| 0.5 | 11 | 132 | 1.091 [0.97, 1.226] | 0.185 |

The alpha x cap surface screen on bb144 z at p = 2e-3: alpha 0.5 is best and wants caps >= 15 (cap 11 1.10, cap 15
1.03); every other alpha is 1.2-3.3x worse (`tuning/data/knob_facts.json: alpha`). The lesson per code: on joint xyz,
low alpha at caps 5-9 beats the reference at every measured p; on bb144 z, low alpha loses at p = 2e-3.

### 4.4 Keep-warm and call cadence

**Final build, shipped keep-warm, machine A** (bb144 xyz a03125c3, single handle, 6 repetitions per condition; keep-warm
every 0.5 ms through a 10 ms busy-wait gap; `tools/results/final-bench/final-bench.json`):

| condition | A-oc p50 / p99 / p99.9 (µs) | A-stock p50 / p99 / p99.9 (µs) | graphics clock at the calls | GPU power |
|---|---|---|---|---|
| gap 0 (back-to-back) | 32.28 / 42.90 / 64.89 | 34.18 / 44.65 / 65.95 | – | – |
| gap 10 µs | 32.29 / 41.40 / 64.99 | 34.20 / 43.29 / 66.30 | – | – |
| gap 10 ms, no keep-warm | 37.02 / 46.18 / 58.24 | 36.64 / 45.80 / 59.20 | OC: 2,677 MHz (71 %), 2,670 (26 %); stock: 2,640 (56 %), 2,647 (41 %) | 41.7 W (OC), 44.0 W (stock) |
| gap 10 ms, keep-warm, last call 1 µs before the decode | **31.72 / 40.61 / 49.32** | **33.80 / 42.91 / 51.25** | OC: 3,187 MHz (96 %); stock: 2,895-2,910 MHz | 50.7 W (OC), 53.0 W (stock) |
| gap 10 ms, keep-warm, last call 25 µs before | 32.26 / 41.11 / 50.49 | 34.27 / 43.03 / 52.31 | OC: 3,187 MHz (97 %) | 50.6 W, 53.0 W |

Keep-warm calls: ~42,000 per condition, 0 errors, 1.50-1.51 µs mean host time per call (max 5.45 µs); output checksums equal
across all conditions. Readings: (1) at a 10 ms cadence keep-warm brings the call back to the back-to-back level or slightly
below it, at +9 W; (2) the overclocked profile loses more to the idle sag (+4.7 µs p50) than stock (+2.5 µs), so
keep-warm matters more there; (3) the last call close to the decode is better than 25 µs ahead (-0.54 µs p50 OC, -0.47
stock); (4) spaced calls have a lower p99.9 than back-to-back ones (no launch-ring stall), and a 10 µs gap already
lowers the p99 by 1.4-1.5 µs. The timed region is the decode call only: a keep-warm call still in flight when a
syndrome arrives delays the decode by up to its ~1.5 µs.

The registry's earlier rule "stop >= 20 µs before the decode" (`tuning/data/knob_facts.json: keep_warm_call`) came from a
bench-side prototype on build 6e; the shipped header says the last call is best placed shortly before the next decode
(`src/core/fastbp_ext.h`), and the final bench agrees.

**Prototype and cadence rows on build 6e** (machine A-oc, a03125c3; `tuning/data/knob_facts.json: keep_warm_call`,
`tuning/data/knob_facts.json: call_cadence`; `tuning/data/knob_facts.json: target_points`):

| condition | p50 / p99 / p99.9 (µs) | note |
|---|---|---|
| gap 0 / 5 µs / 20 µs / 100 µs / 1 ms | 40.56 / 52.34 / 70.89; 40.54 / 51.02 / 70.29; 40.57 / 50.88 / 70.11; 40.65 / 51.14 / 70.81; 41.14 / 52.41 / 69.84 | flat to 100 µs, +0.6 µs p50 at 1 ms; small gaps improve the p99 by 1.3-1.5 µs (the launch ring wraps less often) |
| gap 10 ms (busy-wait / nanosleep) | 46.83 / 62.62 / 94.45; 46.96 / 65.11 / 94.48 | device body 32.74 -> 37.50 µs as the clock drops 3,187 -> 2,677 MHz inside P1; host part 4.36 -> 5.61 µs; nanosleep adds +0.4 p50 / +4 p99 |
| 10 ms, keepalive every 100 µs / 500 µs / 2 ms | 41.85 / 56.64 / 83.14; 41.84 / 56.82 / 86.33; 46.64 / 62.75 / 95.08 | 51.3 W / 50.5 W; at 2 ms the clock reaches only ~2,915 MHz: not enough |
| 10 ms, coarse host sweep (~20 MB) alone / with keepalive 100 µs | 47.57 / 61.71 / 95.59; 42.08 / 53.81 / 80.51 | the sweep alone evicts working lines (p50 +0.3..+0.6) |

**A disagreeing older record.** On an older build on machine F (RTX 5090, driver 575.57.08, CUDA 12.8), the first shot
after a 1-100 ms gap cost +10.0 to +80.1 µs in three runs, mostly host-side, and the GPU stayed in P1 at 2.5-2.9 GHz
for gaps up to 3 s (development record: design record). On A (RTX 5080, driver 615.71.09) the clock sags inside P1 and
keep-warm recovers it. Different card, driver and host: state them with the claim, and measure the idle behaviour on
the deployment machine.

**Identity of the shipped hook** (machine E; development record: validation runs): 5 problem/config cases (bb144 xyz
a03125c3, a05ms31 and `bp.ms30`; bb144 z a05c15; bb288 z a05ms31) x 3 call patterns (before every shot, before every
third shot, 8-call bursts) x 300 shots: all 15 patterns bit-identical to the run without keep-warm in correction,
observables, status and soft output; every call returned 0. The test also checks that a NULL handle and every non-GPU
flag bit return `EINVAL`.

**Machine M (Xeon Silver host, L40S, 64-block plan, a03125c3; `tools/results/host-xeon/host-xeon.json`).** The keep-warm
A/B is paired in one process: both handles are the same build and decode the same shots with the same gap, and only
one handle is kept warm, so the ratio is the keep-warm effect itself (2,000 shots per side; verdicts by the A/A rule
of section 7.1). Shared host: ratios hold, absolute times are indicative.

| condition | bb288 z p50 ratio (verdict) | bb144 xyz p50 ratio (verdict) |
|---|---|---|
| 10 ms gap, keep-warm every 100 µs | 0.9658 (no difference; single measurement) | – |
| 10 ms gap, every 250 µs | -2.6 % in a re-measurement (the first measurement's 0.9663, -3.4 %, did not reproduce) | 0.9764 (no difference) |
| 10 ms gap, every 500 µs | 0.9735 (faster; re-measurement -2.5 %) | 0.9854 (no difference) |
| 1 ms gap, every 500 µs | 0.9914 (faster) | – |
| back to back: the call after a keep-warm against the call after none | p50 1.0046, **p99 1.1267** (70.0 vs 62.1 µs) | p50 0.9996, p99 0.9980 |

Readings: at a 10 ms cadence keep-warm lowers the p50 by about 2.5 % on bb288 z, with no preferred period between
250 and 500 µs (the two differ by 0.1 point); on bb144 xyz the gain (1.5-2.4 %) is not resolved; no p99 effect is
resolved at a cadence (the control bands at p99 are wide); back to back the empty kernel sits ahead of the next decode
on the same stream and can cost its p99, so keep-warm belongs to idle gaps, not to a back-to-back loop. The default
build's own p50 / p99 back to back, at a 10 ms gap without keep-warm and with it (every 500 µs and once 25 µs before
the call; separate sessions) was 62.5 / 77.8, 66.0 / 106.2 and 65.6 / 99.5 µs on bb144 xyz, and 46.9 / 61.1, 50.6 /
106.6 and 49.1 / 104.3 µs on bb288 z. The keep-warm call cost 1.52-1.55 µs of host time as a periodic call and
3.18-3.23 µs as the final call 25 µs before the decode. Output checksums (correction, observables, status) were equal
across all cadence runs, builds and keep-warm modes (84 runs on bb144 xyz, 68 on bb288 z).

**Machine K (RTX 6000 Ada, EPYC host).** No paired keep-warm A/B was run on the RTX 6000 Ada. Its budget sweep has 10 ms
cadence rows (2,000 shots per side, the budget-64 build against the default in each session): the budget's gain holds
there (bb144 xyz p50 ratio 0.7723 without and 0.7732 with keep-warm, back to back 0.7400), and the budget-64 build's own
bb144 xyz p50 / p99 was 64.0 / 83.4 µs back to back, 72.9 / 139.2 µs at 10 ms without keep-warm and 72.6 / 126.2 µs
with it (separate sessions; `tools/results/ada-sm-budget/full-bench.json`, K `full_bench`). Whether keep-warm recovers the
back-to-back level on the deployment board is therefore not measured.

### 4.5 GPU clocks: overclocked against stock

| row (final build, machine A) | A-oc | A-stock | change (computed) | source |
|---|---|---|---|---|
| bb144 xyz a03125c3, p50 / p99 | 32.12 / 42.37 | 34.02 / 44.15 | -5.6 % / -4.0 % | `tools/results/final-bench/final-bench.json` |
| bb144 xyz a0375c7 / a05c11 / a05ms31, p50 | 56.84 / 81.94 / 205.87 | 61.01 / 88.27 / 222.98 | -6.8 % / -7.2 % / -7.7 % | same |
| bb144 z a04375c3 / bb288 z a05ms31, p50 | 23.45 / 126.88 | 24.64 / 137.14 | -4.8 % / -7.5 % | same |
| per-iteration slope | 6.193 µs | 6.734 µs | -8.0 % | computed (section 4.1) |
| legacy base, (0.5, 11) / (0.5, 31) p50 | 319.6 / 739.5 | 330.0 / 771.6 | -3.2 % / -4.2 % (registry: -3.5 %); t_iter 22.0 -> 21.0 µs for +10.4 % core clock | `tuning/data/knob_facts.json: gpu_oc`; `tuning/data/knob_facts.json: bench` |

The iteration is only partly clock-bound (~45 % clock-sensitive on the legacy base, `tuning/data/knob_facts.json: gpu_oc`),
and the gain grows with the cap because the device share grows. The overclock does not stop the idle clock sag
(section 4.4). Validation of outputs under the profile: 12M cross-machine shots and 1.6M same-machine shots with 0
mismatches, 0 Xid over 13.3M overclocked shots (`tuning/data/knob_facts.json: gpu_oc`); the final bench ran 2.6M paired
shots per profile with 0 mismatches and 0 Xid. The registry's note on the profile: "one profile switch per boot (Xid 62)".

### 4.6 SM budget

`DBENCH_GPU_SM_BUDGET` caps the SM count the create path plans grids, capacities and the formulation for (section
2.4). The shipped `ada64` core is the build with budget 64; the device-class table gives it to compute capability 8.9
parts with at least 128 SMs (section 2.3; `docs/BUILD.md`, "Core variants"). Two kinds of rows follow: measurements of
budget builds on real Ada boards, and older proxy rows that cap the grid of a 170-SM part.

**Measured on the Ada class (machines K, L, M).** Method of the K and L sweeps: each candidate is the six core sources
built with one extra `-DDBENCH_GPU_SM_BUDGET=<n>` (sm_89; AVX2 host flags on K, AVX-512 on L). Before any timing an
exactness gate compared correction, observables, status word and soft output per shot against the reference and the
default build (K: 7 builds x 15 rows, five problems at `a03125c3`, `a05c11` and `ms30`, 60 shots of bb144 xyz and 200
of the others per row; L: 5 builds x 15 rows): 0 mismatches. Then paired screens against the default build in one
process: 3 processes x 2 create orders x {main, A/A}, 20,000 shots per side per run (120,000 per side pooled), both
sides on CPU 0, pooled nearest-rank quantiles, the separation rule of section 7.1; the screens' paired outputs had 0
correction and 0 observable mismatches (11.76M shot pairs on K). Ratio = budget build / default build, < 1 is faster.
Sources: `tools/results/ada-sm-budget/rtx6000ada-142sm.json` and `tools/results/ada-sm-budget/rtx4090-128sm.json`
hold the per-build screen rows (120,000 shots per side: ratios, p50 / p99, verdicts) and the exactness flag of each
build; `tools/results/ada-sm-budget/full-bench.json` holds the method counts above (gate builds, rows and shots, shot pairs), the full bench, the
direct pairs, the stage probes, the layout probes, the solo and companion-handle rows and the long-cap follow-up's
A/A bands (`tools/results/ada-sm-budget/README.md`). Column M is the earlier L40S screen as tabulated by the RTX 6000
Ada sweep (`M_earlier_l40s_screen` in the same file): pooled p50 ratios only, a single measurement that was not
repeated.

Pooled p50 ratio at `a03125c3`, bb144 xyz / bb144 z / bb288 z:

| budget | K: RTX 6000 Ada, 142 SMs | L: RTX 4090, 128 SMs | M: L40S, 142 SMs |
|---|---|---|---|
| default against itself | 1.0012 / 1.0049 / 1.0002 | 1.0001 / 1.0000 / 0.9999 | 1.0001 / 0.9994 / 1.0002 |
| 40 | 0.8258 / 0.8795 / 0.8485 | – | 0.6941 / 0.8519 / 0.7219 |
| 48 | 0.7438 / 0.8847 / 0.8893 | 0.7387 / 0.8700 / 0.9576 | 0.6678 / 0.8583 / 0.7798 |
| 56 | 0.7207 / 0.9262 / 0.6661 | 0.7245 / 0.9354 / 0.6812 | 0.6157 / 0.8949 / 0.8442 |
| **64 (`ada64`)** | **0.7393 / 0.9621 / 0.6470** | **0.7404 / 0.9431 / 0.6611** | **0.6564 / 0.9400 / 0.5565** |
| 72 | 0.7709 / 1.0011 / 0.7148 | 0.7850 / 0.9998 / 0.7282 | – |
| 80 | 0.7628 / **1.0613 (slower)** / 0.7214 | – | – |
| geometric mean of the three at budget 64 | 0.7720 | 0.7729 | 0.7003 (computed) |

Every measured budget (40-80 on K, 48-72 on L) was faster than the default on bb144 xyz and bb288 z; the lowest bb144 z
ratio on K is budget 40's (0.8795). Null controls: bb72 z and hgp98 z were no different at every budget on both boards
(their in-place launches, 32 x 256 and 31 x 256, do not change with the budget). Other shapes on K at budget 64:
hgp98 xyz 0.5805, sc9 z 0.6135 (faster). At budget 64 the gain is smaller on the RTX 6000 Ada (a 300 W board with an
EPYC host) than on the L40S (350 W, Xeon host) on all three problems; the records do not separate the causes. The
direction is the same on all three boards.

Budget 56 against 64. On K the two are within noise of each other: the direct paired full bench (240,000 shots per
side, back to back) gave p50 0.9853 on bb144 xyz and 0.9596 on bb144 z, both "no difference" by the A/A rule (only its
10 ms cadence rows separate); the direct screen favours 64 on bb288 z (56 / 64 = 1.0344, slower); and at p99.9 the
ratios of 64's back-to-back full rows against the default are lower than 56's on all three problems (in absolute
terms 64's p99.9 is the higher one on bb144 xyz and bb144 z). On L the direct pair favours 64 (geometric
mean of 56 / 64: 1.0204). The K sweep's own pick by geometric mean was 56 (0.7633 against 0.7720, a 1.15 % gap), which
the direct pairs do not separate: the shipped `ada64` (64) stands and no 56 variant is built.

Full bench, budget 64 against the default build, back to back (160,000 shots per side; `tools/results/ada-sm-budget/full-bench.json`; absolute times are K's EPYC host with the AVX2 core and L's VM guest, so indicative only). The
per-build screen summaries hold the 120,000-per-side screens of the same builds, which agree within 0.2 µs at the p50
and within 1.3 µs at the p99 (largest: L bb144 z, 39.02 / 46.07 against 41.38 / 48.20 µs, ratio 0.9431 / 0.9557):

| problem | K p50 / p99 (µs), 64 against default | K ratio p50 / p99 | L p50 / p99 (µs), 64 against default | L ratio p50 / p99 |
|---|---|---|---|---|
| bb144 xyz | 64.0 / 83.4 against 86.5 / 106.4 | 0.7400 / 0.7834 | 61.6 / 74.4 against 83.1 / 96.0 | 0.7409 / 0.7752 |
| bb144 z | 43.9 / 51.3 against 45.7 / 53.2 | 0.9592 / 0.9643 | 38.9 / 44.8 against 41.2 / 47.1 | 0.9443 / 0.9505 |
| bb288 z | 50.9 / 66.0 against 78.4 / 95.5 | 0.6487 / 0.6915 | 47.2 / 58.7 against 71.5 / 95.2 | 0.6604 / 0.6170 |

At a 10 ms cadence the ratios hold (K, bb144 xyz: 0.7723 without keep-warm, 0.7732 with it; section 4.4).

**Long caps** (budget 64 against the default build, the same screen, 120,000 shots per side):

| problem | preset | K p50 / p99 ratio (verdict) | K p50, 64 / default (µs) | L p50 / p99 ratio (verdict) |
|---|---|---|---|---|
| bb144 xyz | `a05c11` | **1.0901 / 1.0776 (slower / slower)** | 147.5 / 135.3 | **1.0507 / 1.0582 (slower / slower)** |
| bb144 xyz | `ms30` | **1.3725 / 1.3308 (slower / slower)** | 350.4 / 255.3 | **1.2847 / 1.2846 (slower / slower)** |
| bb144 z | `a05c11` | 0.9917 / 0.9815 (no difference / faster) | 75.5 / 76.2 | 0.9551 / 0.9638 (faster / faster) |
| bb144 z | `ms30` | 0.9936 / 1.0062 (no difference) | 159.1 / 160.1 | 0.9623 / 0.9765 (faster / faster) |
| bb288 z | `a05c11` | 0.7593 / 0.7492 (faster / faster) | 85.2 / 112.2 | 0.7404 / 0.7508 (faster / faster) |
| bb288 z | `ms30` | 0.8610 / 0.8723 (faster / faster) | 174.0 / 202.1 | 0.8640 / 0.8550 (faster / faster) |
| hgp98 xyz | `ms30` | 0.5901 / 0.6207 (faster / faster) | 177.4 / 300.6 | not measured |

K's two bb144 xyz rows and its three other `ms30` rows are a follow-up with the sweep's own scripts and builds on the
same machine (`follow_up` in the K file). During the last three of them another tenant raised the host's 1-minute
load average to 9.5-18.3 (`long_cap_follow_up` in `tools/results/ada-sm-budget/full-bench.json`); their A/A bands stayed within +-0.8 %, so the verdicts stand and their absolute times are
indicative. On L the create-path probe at the long presets gives the same launch and formulation as at `a03125c3` (12
records, 0 differences).

Readings. (1) The measured slow cells are bb144 xyz at `a05c11` and `ms30`, on both boards and more strongly on the
142-SM board; it is not "joint problems" in general (hgp98 xyz at `ms30` is 0.59 with budget 64), and bb144 z and bb288
z stay level or faster. (2) No rule keyed on the preset alone is a clean win on every measured problem on both boards:
at `ms30` bb144 xyz wants the default build while bb288 z and hgp98 xyz want 64, and at `a05c11` bb144 xyz wants the
default while bb288 z wants 64. The shipped table is therefore keyed on the device only: set
`ELDPC_DEVICE_CLASS=default` for long-cap presets on large joint problems such as bb144 xyz and for comparisons at
stock-equivalent settings (`ms30`), or time both variants (`docs/BUILD.md`, "Core variants"; section 2.3).

**Why the 64-block plan wins at cap 3 and can lose at long caps.** A stage-probe build of the same sources (never
shipped) stamps `%globaltimer` at kernel entry and at the status publish, and the driver records the interval from the
launch call's return to the host observing the status; bb144 xyz, `a03125c3` (`tools/results/ada-sm-budget/full-bench.json`, `stage_probe`):

| machine | build | device body p50 (µs) | launch return to observed p50 (µs) | outside the body p50 (µs) |
|---|---|---|---|---|
| K | default, 142 blocks | 38.91 | 80.79 | 42.13 |
| K | budget 64 | 40.96 | 54.85 | 13.79 |
| L | default, 128 blocks | 34.82 | 81.73 | 46.92 |
| L | budget 64 | 37.89 | 52.96 | 15.11 |

(K: 4 processes x 1,000 shots per build. On L's bb288 z the body falls with budget 64, 28.67 -> 22.53 µs, and the part
outside it 50.31 -> 19.88 µs.) The saving is in the launch and observation interval (-26 to -29 µs on bb144 xyz),
while the device body of bb144 xyz grows by 2.05 µs on K and 3.07 µs on L at cap 3 (computed differences): each
iteration runs on 64 blocks instead of 142 (128 on L). The records read the long-cap loss as that per-iteration growth
times the cap outweighing the fixed saving; the per-iteration slope of the two variants was not measured separately.
On K the 64-block plan of bb144 xyz also runs without the b2c window (below).

Create-time shapes on K (layout-probe builds of the same sources, create only; `tools/results/ada-sm-budget/full-bench.json`, `layout_probe`; opt-in shared memory per block 101,376 B on K and L, as on the RTX 5080):

| problem | default build, 142 SMs | budget 64 |
|---|---|---|
| bb144 xyz | 142 x 512, fused summary, b2c window 9 rounds / 48,384 B | 64 x 512, fused summary, no b2c window |
| bb144 z | 117 x 256, in place | 64 x 512, fused summary, window 9 rounds / 13,824 B |
| bb288 z | 142 x 512, fused summary, window 9 rounds / 16,128 B | 64 x 512, fused summary, window 9 rounds / 36,864 B |
| hgp98 xyz | 245 x 256, in place | 64 x 512, fused summary, window 9 rounds / 23,040 B |
| sc9 z | 142 x 512, fused summary, window 9 rounds / 9,216 B | 64 x 512, fused summary, window 9 rounds / 20,736 B |
| bb72 z, hgp98 z | 32 x 256 and 31 x 256, in place | the same |

The other budgets' shapes (the same layout probes) match the ratio table: on bb288 z budget 48 falls back to the
in-place 144 x 256 schedule
(0.8893) while 56 and 64 keep the fused summary (0.6661, 0.6470); at budget 40 bb144 xyz loses the row-slot form of the
fused kernel (0.8258 against 0.7207 at 56); at 72 bb144 z keeps the default's 117 x 256 in-place launch (1.0011, no
difference) and at 80 it takes the fused summary and is slower (1.0613). L's default launches are 128 x 512 (bb144 xyz
and bb288 z, summary) and 117 x 256 (bb144 z, in place).

**Paired ratios of two launch plans are conservative** (`tools/results/ada-sm-budget/full-bench.json`, `companion_handle` and
`solo_one_library_per_process`). Both builds
of a paired session share one process, and on L a
64-block build made the default side faster than it is against itself on the summary problems (default side against
budget 64 compared with its own A/A block: bb288 z -9.7 %, bb144 xyz -3.2 % at `a03125c3`; with `decoder=bp` bb288 z
-15.1 %, so the effect is device-side, not the host OSD path). One library per process (alternating, 3 rounds x 20,000
shots) gave bb144 xyz 0.6567, bb144 z 0.9390 and bb288 z 0.5877 against the paired 0.7409, 0.9443 and 0.6604. Order two
close budgets by a direct pair, not by their separate default-relative ratios (section 6.6).

**Proxy rows (older build, machine F).** The grid of a 170-SM RTX 5090 capped at 84 blocks to emulate an 84-SM part;
a proxy, not a measurement of an 84-SM GPU (development record: design record).

| measurement | 84-block shape | shipped 170-SM grid | cost |
|---|---|---|---|
| per iteration, bb144 z in place | 6.11 µs (84 x 256) | 4.78 µs (117 x 256) | +28 % |
| per iteration, bb288 z in place | 11.42 µs (84 x 256) | 6.34 µs (342 x 256) | +80 % |
| per iteration, bb144 xyz row summary | 19.50 µs (84 x 512) | 14.39 µs (170 x 512) | +35 % |
| `bp.ms30` whole call, bb288 z / bb144 xyz | 223.5 / 599.8 µs | 198.8 / 439.6 µs | +12 % / +36 % |
| grid request caps | 252 in place / 84 summary | 510 in place / 170 summary | from the occupancy query |

Exactness: 70-SM and 64-KiB-L2 budget builds of the final code are bit-identical on all 11 recorded problems at four
configs (60 shots each), and 88 cap-identity rows are bit-identical (development record: design record). The b2c
window is the sharpest SM-count effect: it fits at 84 blocks and at 170, and a 70-SM part loses it (section 2.5). The
Ada budget builds above (40-80 on K, 48-72 on L) passed the gate described with them.

**The primary class from the rules, checked on K.** With 142 SMs and 96 MiB L2, 8 x nnz(H) of bb144 xyz (3,511,296 B)
is 28.7x below the L2 (computed), so the L2 term never fires on the recorded problems; bb144 xyz takes the fused
summary at 142 and at 64 blocks; bb144 z's in-place grid of 117 blocks fits 142 SMs, so the default build keeps it
in place, as the K probes show. A new problem's shape must still be verified on the part (section 2.5).

### 4.7 L2 budget

`l2 = min(device L2, DBENCH_GPU_L2_BUDGET)`; the summary formulation is forced when `8 x nnz(H) > l2`
(`src/core/bp_gpu.cu`). Margins on the recorded problems: bb144 xyz 3.35 MiB of messages (19.1x below 64 MiB), bb288 z
0.82 MiB (78.3x). A 64-KiB L2 build is bit-identical (60 shots bb144 xyz, 300 bb144 z, 300 bb288 z), changes the
formulation only of the z problems (bb144 xyz already uses the summary) and still takes the b2c window (development
record: design record; `tuning/data/knob_facts.json: b2c_window`). The budget emulates a smaller-L2 part on a larger one; it
never changes the arithmetic.

### 4.8 CUDA toolkit

| effect | value | machine | source |
|---|---|---|---|
| 13.3 vs 13.0, same source (one build step) | cap 3 p50 -1.8 µs with 13.3 vs -0.1 µs with 13.0; two-point slope ratio 0.9116 vs 0.958 | D | `tuning/data/knob_facts.json: cuda_toolkit` |
| why | 13.0 spills in the record kernels (REG 128, STACK 40 B, 68 STL/LDL sites); 13.3 has an 8 B frame and two kernels drop to 127 registers | D | same |
| 13.2 | two-point slope ratio 0.8884 for another build step | D | same |
| 12.8 + driver 575 | cannot run CUDA 13 code; such machines can check exactness of 12.8 builds, never time | – | same |

The toolkit changes the kernels enough to reorder build decisions (a 5 % per-iteration difference above). Re-run
exactness and latency after any toolkit change; never mix rows from two toolkits; print `nvcc --version` and
`build_info` with every row.

### 4.9 Host ISA

| effect | AVX2 | AVX-512 | machine | source |
|---|---|---|---|---|
| observables base scan of an older build | 8.97-8.98 µs | 4.62-4.66 µs | H (Zen 5) | `tuning/data/knob_facts.json: observables_block_scan` |
| host OSD-0 order filter, reference engine | – | the AVX-512 reference engine was 16-19 % faster than AVX2 | I (Zen 4 V-Cache) | `tuning/data/knob_facts.json: order_filter` |
| posterior hand-off step | same gain on both (-14.4..-17.8 µs p50) | same | H | `tuning/data/knob_facts.json: posterior_one_batch_handoff` |
| exactness | bit-identical | bit-identical | all | `tests/test_isa.py`; `tuning/data/knob_facts.json: observables_block_scan` |

The ISA matters for the host OSD walk and the observables on wide problems; GPU terms are unaffected.

**Whole call on a Xeon host (machine M, final core with the 64-block launch table, `a03125c3` unless stated;
`tools/results/host-xeon/host-xeon.json`).** Seven host builds passed the exactness gate first (five problems x three
presets, 2,700 shots per comparison, against the reference and the default AVX-512 build: correction, observables,
status and soft output identical). Paired screens against the shipped AVX-512 flag set (`-mavx2 -mavx512f -mavx512vl
-mavx512dq -mavx512bw`, with `-ffp-contract=off`), 120,000 shots per side, CPU 0 with an idle sibling:

| build against the shipped AVX-512 set | result | rows |
|---|---|---|
| AVX2 (`-mavx2`) | slower: screens p50 1.0003-1.0698 (slower on 5 of 6 rows), p99 1.0137-1.0535 (6 of 6); back-to-back full rows (160,000 shots per side) bb144 xyz p50 / p99 1.0370 / 1.0556, bb288 z 1.0679 / 1.0491, bb144 z 1.0254 / 1.0261. A re-measurement with the same command and pinned CPU gave bb144 xyz 1.039 / 1.076 and bb288 z 1.111 / 1.134: quote **AVX2 about 4-13 % slower** on these two problems at `a03125c3` | bb144 xyz, bb144 z, bb288 z, hgp98 xyz at `a03125c3`; bb144 z `a05c15`; bb288 z `a05ms31` |
| AVX2 where the GPU share is larger | bb144 z `a05c15` p50 1.0003 (no difference), p99 1.0137; bb288 z `a05ms31` p50 1.0204; on the default 142-block launch table about half the 64-block effect (bb144 xyz p50 1.0193, bb288 z 1.0307): a longer GPU call dilutes the host term | same screens; the two 142-block builds were screened against each other |
| AVX2 with `decoder=bp` (no OSD-0) | bb144 xyz p50 1.0092 / p99 1.0091, against 1.0363 / 1.0535 with OSD-0: most of the ISA effect is in the OSD-0 elimination, a small part in the host work both paths share | screens |
| `-march=sapphirerapids` | no difference on 10 of 10 p50 / p99 comparisons (p50 0.9980-1.0001) | 5 screen rows; its 5 full rows (back to back and 10 ms cadence) also no difference |
| `-march=x86-64-v4` | no difference on 10 of 10 (p50 0.9981-1.0028) | 5 screen rows |
| clang++ 16, the same AVX-512 flags | no gain: slower on 1 of 6 p50 / p99 comparisons, no difference on 5 | 3 screen rows |

So on an AVX-512 Xeon keep the automatic AVX-512 core and the shipped flag set: no `-march` flag and no other compiler
has a measured reason (`tools/results/host-xeon/host-xeon.json`). The host above is Emerald Rapids, the same AVX-512 flag family as Sapphire Rapids.

**Whole call on a Sapphire Rapids-W host (machine N: Xeon w9-3495X with an L40S; the shipped package at `21b2820`,
`ada64` core, `a03125c3`, back to back; `tools/results/host-sprw/summary.json: isa_ab`).** `ELDPC_ISA` is read when
the library loads, so each core was timed in its own bench.py processes: 4 full-length single-config runs per core in
the order AVX-512, AVX2, AVX2, AVX-512, repeated (6 sessions x 1500 calls per handle; the value of a run is the pooled
statistic of its handle A, 9,000 timed calls), each with its own A/A control, pinned to CPU 8. The difference is the
mean over the AVX2 runs minus the mean over the AVX-512 runs; "run spread" is the larger range of one core's run
values; "A/A band" is the largest end of any run's own A/A band (the range of its per-session A/A differences):

| problem, call | AVX-512 p50 / p99 (µs) | AVX2 p50 / p99 (µs) | AVX2 - AVX-512 p50 / p99 (µs; ratio) | run spread p50 / p99 (µs) | A/A band p50 / p99 (µs) |
|---|---|---|---|---|---|
| bb144 xyz, C | 59.25 / 69.57 | 60.20 / 71.10 | +0.95 / +1.53 (1.016 / 1.022) | 1.69 / 5.92 | 1.57 / 6.68 |
| bb144 xyz, Python | 67.42 / 117.70 | 68.85 / 121.43 | +1.43 / +3.74 (1.021 / 1.032) | 0.71 / 5.39 | 1.60 / 34.00 |
| bb288 z, C | 43.24 / 50.24 | 45.31 / 53.48 | +2.07 / +3.24 (1.048 / 1.065) | 0.28 / 0.94 | 0.97 / 4.67 |
| bb288 z, Python | 51.93 / 100.29 | 53.90 / 102.69 | +1.97 / +2.40 (1.038 / 1.024) | 0.42 / 1.65 | 0.90 / 40.18 |

Readings. (1) On bb288 z the AVX2 core is slower by about 2 µs at the p50 in both calls, outside the run spread and
the A/A band. (2) On bb144 xyz the C p50 difference is inside both, so it is not resolved; the Python p50 difference
separates the runs of the two cores but lies inside the A/A band. (3) Every p99 difference lies inside the A/A band.
The sizes are smaller than machine M's 4-13 % above (paired in-process screens of harness builds on an Emerald Rapids
host; the records do not separate the causes); the direction is the same, so keep the automatic AVX-512 core. On the
deployment host itself the same A/B (section 7) gives the size there.

### 4.10 Thread placement on a two-socket Xeon host

Machine M, the default AVX-512 build with the 64-block plan, `a03125c3`, timed A/A in one process under five pinning
conditions (`tools/results/host-xeon/host-xeon.json`). Each session: one discarded warm-up process, then 4 processes x
both create orders x 20,000 shots per side (320,000 pooled); two rounds in opposite order, each round's ratios taken
against its own idle-sibling row, so host drift is not read as a pinning effect. These are between-session ratios, not
a paired A/B; the ranges span the two rounds.

| condition | bb144 xyz p50 / p99 / p99.9 ratio | bb144 z p50 / p99 / p99.9 ratio |
|---|---|---|
| CPU 0 on the GPU's NUMA node, SMT sibling idle (reference) | 1 (61.7 / 76.0-76.3 / 129.8-131.9 µs) | 1 (42.5 / 47.8-48.0 / 63.0-63.1 µs) |
| CPU 32, the second hardware thread of the same core, sibling idle | 1.0019-1.0038 / 0.9932-1.0086 / 0.9621-0.9970 | 1.0023-1.0026 / 1.0061-1.0079 / 0.9940-0.9945 |
| CPU 0 with a busy loop on its SMT sibling | 1.0329-1.0334 / 1.0744-1.0761 / 1.1555-1.1669 | 1.0624-1.0638 / 1.0878-1.0913 / 1.1412-1.1425 |
| CPU 16, the other socket (NUMA node 1) | 1.0051-1.0065 / 0.9955-0.9972 / 0.9582-0.9684 | 1.0106-1.0232 / 1.0146-1.0641 / 0.9785-1.0091 |
| unpinned | 1.0065-1.0113 / 1.0222-1.0460 / 0.9935-1.0268 | 1.0823-1.1111 / 1.1363-1.1515 / 1.1128-1.2499 |

Readings: pin the calling thread to a core on the GPU's NUMA node and keep that core's SMT sibling idle (a busy
sibling costs +3.3 to +6.4 % p50, +7.4 to +9.1 % p99 and +14.1 to +16.7 % p99.9); either hardware thread of the core
is equivalent; unpinned costs +0.7 to +11.1 % p50 and +2.2 to +15.1 % p99, most on the in-place bb144 z shape; the
other socket costs +0.5 to +2.3 % p50 and its p99 is not resolved by these sessions. `taskset -c <cpu>` is enough
(`numactl` is not needed). The deployment class's single-socket Xeon W has no other socket; the sibling and pinning
rows are the ones that carry over as directions, their sizes are M's.

### 4.11 The CPU core

`Decoder(..., device="cpu", threads=N)` runs the CPU core (`src/cpu/`): the multi-threaded exact min-sum BP engine
with the same bit-packed OSD-0 engine as the GPU core, bit-identical to the reference and to the GPU core (`README.md`,
"CPU core"; `tools/results/cpu/exactness*.json`). Its knobs are `threads` (the threads that compute one shot, the calling
thread included; the started threads busy-wait between shots) and `cpus` (their pinning). All rows: machine E (Ryzen 9
7950X3D: CCD 0 with 96 MiB L3 = CPUs 0-7 and SMT siblings 16-23, CCD 1 with 32 MiB L3), a shared machine with
`powersave` governor, so every row is a screen; same-session A/B rows decide (development record: CPU-core study;
the packaged-wheel rows: `tools/results/cpu/screen.json`, `tools/results/cpu/placement.json`).

| measurement | result | source |
|---|---|---|
| CPU core against the GPU core on the same machine, bb144 xyz `a03125c3`, `threads=8` on CCD 0, AVX-512 (one GPU-locked session, 4 processes x 2,000 shots) | p50 / p99 / p99.9 223.5 / 288.5 / 370.9 µs against 36.1 / 47.7 / 89.6 µs: 6.19x / 6.05x / 4.14x | `tools/results/cpu/gpu_same_machine.json` |
| the installed wheel, same cell | `threads=8` 230.4 µs p50 against the GPU core's 42.0 µs in its session (5.48x); `threads=1` 1,180.3 µs; the packaged core within 0.5 % (p50) and 0.8 % (p99) of the same sources rebuilt with the same flags | `tools/results/cpu/screen.json` |
| `threads` (same-session A/B, CCD 0) | `threads=8` against 6: p50 0.781; against 4: 0.632 | development record: CPU-core study |
| placement | `threads=16` across both CCDs 2.41x the p50 of `threads=8` on CCD 0 (the wheel's own screen: 2.334x p50, 2.008x p99); `threads=10` and 12 spanning both CCDs 2.20x and 2.11x; `threads=8` on the SMT second contexts 1.001x-1.006x, `threads=16` using them on one CCD 1.017x (no gain) | development record: CPU-core study; `tools/results/cpu/placement.json` |
| host ISA | the AVX-512 CPU core against its AVX2 build at `threads=8`: p50 0.8686 on CCD 0 and 0.8550 on CCD 1 (13.1 % and 14.5 % faster) | development record: CPU-core study |
| preset cost, `threads=8`, AVX-512 | `a05c11` 3.525x and `ms30` 9.557x the `a03125c3` p50 (781.6 and 2,119.1 µs); cap 1 0.416x and cap 2 0.761x of cap 3 (explorations, not presets) | development record: CPU-core study |
| OSD-0 share (observable-free twins, k = 0, BP+OSD-0 against plain BP on the same shots) | bb144 xyz 15.1 of 223.0 µs (6.8 %); bb288 z 16.7 of 66.9 µs (25.0 %): BP dominates the CPU call | development record: CPU-core study |
| other problems, `a03125c3`, `threads=8`, CCD 0 | p50 bb144 z 30.1 µs, bb72 z 15.2 µs, bb288 z 66.1 µs, hgp98 z 17.4 µs | development record: CPU-core study |

Readings: keep a CPU decoder's threads and its calling thread on one CCD (one L3) or one socket and on physical cores;
spanning CCDs loses more than the extra cores buy at every count measured between 8 and 16. The CPU core is a
fallback on bb144 xyz (about 6x the GPU core's latency on the same AVX-512 machine; about 10x on a Zen 2 AVX2 host with
an RTX 6000 Ada, `tools/results/cpu/zen2-avx2-screen/`). There is no same-session GPU row for the
small z problems, so whether the CPU core is competitive there is not measured; compare on the deployment machine. A
two-socket or Xeon host has no CPU-core row: the one-socket advice is an expectation from the one-CCD result
(`docs/SUPPORT.md`, "The CPU core").

---

## 5. Accuracy against speed

### 5.1 The rules

| rule | statement | source |
|---|---|---|
| both matter | lower LER is better and lower latency is better; which wins depends on the magnitudes (how much LER for how many µs) and on the deployment (required latency, tolerable LER, how many decodes run in parallel) | policy |
| the baseline ("stock") | ldpc BP+OSD-0 at its common settings, min-sum alpha 0.625, 30 iterations, OSD-0, on the CPU. eldpc's `ms30` is bit-identical to it, so `ms30` has stock's LER; it is a parity check, not a separate comparison row. Tuned settings, including the (0.5, 31) reference, are not stock | `tuning/data/knob_facts.json: target` (stock_config); section 0.1 |
| acceptable | the configuration's LER is cleanly below stock's at the same noise point: its one-sided upper bound clears the bar. On bb144 xyz at p = 1e-3 stock is 2.2e-4, the bar was 1e-4, tested as a one-sided 99 % Clopper-Pearson bound below 1e-4 | `tuning/data/knob_facts.json: target` (ler_bar); `tuning/data/knob_facts.json: alpha` (3M-shot confirmation) |
| prefer the fastest | among acceptable configurations take the fastest on the deployment machine; accept a lossier one only for a worthwhile speed gain | policy |
| show the trade | present a Pareto table (latency against LER, µs saved per LER factor) per noise point, not an equivalence test | policy |
| more than one noise point | when the choice depends on it, check the bar at every noise point the deployment must cover; the cap's cost grows with p | `tuning/data/knob_facts.json: cap` |
| size for the bar | size an LER campaign for "clearly below the bar", which needs far fewer shots than a tight ratio | policy |
| exactness first | no build or option that changes any output relative to the reference build of the same sources | section 0.1 |
| comparison rows | on the same shots and machine, single-shot p50 / p99 and LER: stock on the CPU; optionally the same common settings on another GPU BP+OSD implementation (for example NVIDIA CUDA-Q QEC) and that implementation at the tuned config; eldpc at its tuned config | policy |

### 5.2 The stock baseline as measured

| row | latency p50 / p99 | LER at p = 1e-3 | condition | source |
|---|---|---|---|---|
| stock on bb144 xyz, ldpc 2.4.1 Python call, one core of machine A's host | 72.8 / 143.0 ms | 2.2e-4 (174 / 800,000) | 7,103 shots | latency: `tools/results/stock-baseline/stock-baseline.json`; LER: `tuning/data/knob_facts.json: ler_reference` |
| stock on bb144 xyz, the ldpc reference through the C ABI, machine A's host in bench mode | 65.0 / 116.7 ms | same | an earlier session | `tuning/data/knob_facts.json: bench` (cpu) |
| stock on bb144 z, ldpc 2.4.1, machine A's host | 7.11 / 7.67 ms | 5.9e-5 | 19,998 shots | `tools/results/stock-baseline/stock-baseline.json` |

Two stock latencies on the same host (65.0 vs 72.8 ms p50): Python call vs C ABI and different sessions. Against them
a03125c3 on A-stock (34.02 µs) is 1,900-2,140x faster at ~23x lower LER (computed: 65.0 ms and 72.8 ms over 34.02 µs;
2.2e-4 / 9.3e-6).

### 5.3 Pareto table, bb144 xyz

Latency: final build on machine A (section 4.1; for configs that are not presets, the alpha 0.5 row at the same cap,
since alpha moves latency by < ~1 µs). LER ratios against (0.5, 31) on paired shots (sections 4.2, 4.3). There is no
stock LER row at p = 2e-3 or 3e-3: at 1e-3 the reference is 17x below stock; measure stock at your noise point before
calling a ratio near 1 acceptable there.

| config | alpha | cap | A-oc p50 / p99 (µs) | A-stock p50 / p99 (µs) | p = 1e-3 | p = 2e-3 | p = 3e-3 |
|---|---|---|---|---|---|---|---|
| stock (ldpc, CPU) | 0.625 | 30 | 65.0-72.8 ms p50 (CPU) | same | LER 2.2e-4 | no row | no row |
| `a03125c3` | 0.3125 | 3 | 32.12 / 42.37 | 34.02 / 44.15 | LER 9.3e-6 pooled; 1.17e-5 (CP99 1.71e-5); ratio 0.591 | 0.85 / 0.887 / 0.923 (upper 0.990) | **1.066 (upper 1.094) / 1.074: worse than the reference** |
| `a04375c3` | 0.4375 | 3 | not timed on bb144 xyz; cap 3 rows: 32.12-32.54 | 34.02-34.44 | acceptable per the 3M-shot confirmation | no row | no row |
| (0.375, 5) | 0.375 | 5 | 44.74 / 57.11 | 47.75 / 60.15 | acceptable per the 3M-shot confirmation | 0.63 [0.554, 0.719] / 0.728 | 0.795 [0.763, 0.828] |
| `a0375c7` | 0.375 | 7 | 56.84 / 67.88 | 61.01 / 72.20 | 0.742 [0.59, 0.938] (10M) | 0.647 / 0.622 | 0.724 / 0.7283 (upper 0.7497) |
| (0.3125, 7) | 0.3125 | 7 | 57.20 / 69.95 | 61.32 / 74.06 | 0.864 [0.503, 1.5] (2M) | 0.661 / 0.649 | 0.707 [0.674, 0.742] |
| (0.375, 9) | 0.375 | 9 | 69.54 / 82.46 | 74.78 / 87.61 | – | 0.611 [0.536, 0.698] | 0.71 [0.679, 0.741] |
| `a05c11` | 0.5 | 11 | 81.94 / 95.11 | 88.27 / 101.26 | 0.935 [0.813, 1.08] (10M) | 1.114 / 1.049 / 0.986 | 1.027 [1.001, 1.054] |
| `a05c15` | 0.5 | 15 | 106.71 / 119.58 | 115.13 / 128.25 | 0.971 [0.819, 1.15] (5M) | no row | no row |
| `a05ms31` | 0.5 | 31 | 205.87 / 219.71 | 222.98 / 237.22 | reference, 1.28e-5 | 1.0 | 1.0 |
| `ms30` | 0.625 | 30 | no final-build row | – | = stock | = stock | = stock |
| `ms100`, `ms30a075` | 0.625 / 0.75 | 100 / 30 | no row | – | no record | no record | no record |

Sources: `tools/results/final-bench/final-bench.json` (latency); `tuning/data/knob_facts.json: alpha`, `tuning/data/knob_facts.json: cap`,
`tuning/data/knob_facts.json: ler_reference`; `tools/results/stock-baseline/stock-baseline.json` and development record: validation runs (LER).

### 5.4 The frontier per noise point (bb144 xyz, A-oc p50)

Latency from section 4.1 (`tools/results/final-bench/final-bench.json`); LER ratios from sections 4.2-4.3 (knob facts and
validation runs).

| noise point | frontier, fastest first | dominated or failing | reading |
|---|---|---|---|
| p = 1e-3 | `a03125c3` (32.1 µs; LER 9.3e-6, ~23x below stock) | every slower config: its LER is within noise of `a03125c3`'s | take `a03125c3`; nothing measured buys LER at this p |
| p = 2e-3 | `a03125c3` (32.1 µs; 0.85-0.92) -> (0.375, 5) (44.7 µs; 0.63-0.73) -> (0.375, 9) (69.5 µs; 0.611) | `a0375c7` (56.8 µs) is within noise of (0.375, 5); `a05c11`, `a05c15`, `a05ms31` are slower and no better | +12.6 µs buys ~1.4x lower LER |
| p = 3e-3 | (0.375, 5) (44.7 µs; 0.795) -> (0.3125, 7) / `a0375c7` (56.8-57.2 µs; 0.707-0.728) | `a03125c3` is worse than the reference (1.066-1.074) and 1.46x worse than `a0375c7` on the same shots; (0.375, 9) buys nothing over cap 7; `a05c11` is significantly worse than the reference | the default is not the choice when p reaches 3e-3 |

The frontier is a statement about the recorded points only. If the deployment must cover p up to 3e-3, cap 5-7 at
alpha 0.3125-0.375 is the measured frontier; `a03125c3` is the choice only where p <= 2e-3 is guaranteed. The package
default `a03125c3` (`python/eldpc/__init__.py`) is the confirmation stage's fastest acceptable configuration at
p = 1e-3 and 2e-3 (3M-shot confirmation); earlier records used alpha 0.5 and then cap 11 as defaults
(`tuning/data/knob_facts.json: alpha`, `tuning/data/knob_facts.json: cap`). Treat the default as a starting point for
a new circuit, not as an acceptance.

### 5.5 bb144 z and other codes

Latency: final build on machine A, single handle, 4 processes; per class = converged / OSD-path p50 at A-oc
(`tools/results/final-bench/final-bench.json`).

| problem / config | A-oc p50 / p99 (µs) | A-stock p50 / p99 (µs) | per class (µs) | p = 1e-3 | p = 2e-3 vs (0.5, 31) | source (LER) |
|---|---|---|---|---|---|---|
| bb144 z, stock (ldpc, CPU) | 7.11 / 7.67 ms | same | – | 5.9e-5 | no row | `tools/results/stock-baseline/stock-baseline.json` |
| bb144 z `a04375c3` | 23.45 / 33.24 | 24.64 / 34.48 | 22.46 / 23.95 | 5.75e-6 (23 / 4.0M; 99 %: 3.1e-6 to 9.6e-6), ~10x below stock | not measured at cap 3; (0.4375, 7) 1.504x | `tools/results/stock-baseline/stock-baseline.json`; `tuning/data/knob_facts.json: alpha` |
| bb144 z `a05c11` | – | – | – | no row | 1.091 [0.97, 1.226] | `tuning/data/knob_facts.json: cap` |
| bb144 z `a05c15` | 61.50 / 75.76 | 65.56 / 80.72 | 54.12 / 67.02 | no row | cap 15 1.03 (alpha x cap surface screen) | `tuning/data/knob_facts.json: alpha` |
| bb144 z `a05ms31` | 61.60 / 133.50 | 65.53 / 143.82 | 54.31 / 124.39 (bimodal) | no row | reference | – |
| bb288 z `a05ms31` | 126.88 / 138.52 | 137.14 / 148.98 | 58.20 / 127.37 | no row | no row | – |

On bb144 z, `a04375c3` is acceptable at p = 1e-3 with a 10x margin, but low alpha loses 1.5-3.5x to the reference at
2e-3; above 1e-3 measure before using it, and fall back to alpha 0.5 with caps >= 15. For any new code the surfaces
above are a starting grid, not an answer.

### 5.6 Heralded-loss problems

**The construction first.** `HERALDED_ERASE` on t qubits writes t measurement records, so every later `DETECTOR` or
`OBSERVABLE_INCLUDE` target `rec[-k]` with k > t must become `rec[-(k+t)]`. An insertion that copies those targets
unchanged makes some final detectors read herald records; the development corpus's loss problems were built that
way. The package's verify suite applies the shift (`verify/eldpc_verify/circuits.py`, `insert_final_herald`;
`tests/verify/test_herald.py`; `docs/VERIFY.md`); `tuning/` and `bench/` insert no heralds. With the shift, the
circuit's DEM at p_erase = 0 equals the clean circuit's (flattened) DEM coefficient for coefficient. Loss here is
final-measurement loss only; mid-circuit loss is not covered. On the uncorrected bb72 problem every decoder looked
3.6-15x worse (a03125c3 3.31e-3, a05ms31 1.55e-3, ms30 1.78e-3, 100,000 shots each) and the ranking changed: do not use
an LER of a loss problem built without the shift.

**LER on the corrected problems** (`tools/results/heralded-loss/heralded-loss.json`; machine E, harness build of the final
core at `a03125c3` and `a05ms31`, and the ldpc 2.4.1 reference at `ms30`, i.e. stock settings, decoding the same shot
stream per problem; each run stopped at 100 failures, at least 100,000 and at most 2,000,000 shots; 95 %
Clopper-Pearson intervals; no paired test):

| problem (m x n) | p_erase | `a03125c3` LER (failures / shots) [95 % CI] | `a05ms31` | stock (`ms30`) | reading |
|---|---|---|---|---|---|
| `bb72_unif_p1e-3_r6_z_loss1e-2` (324 x 2,376) | 1e-2 | 2.28e-4 (114 / 500k) [1.88e-4, 2.74e-4] | 1.05e-4 (105 / 1M) [8.59e-5, 1.27e-4] | 4.93e-4 (148 / 300k) [4.17e-4, 5.79e-4] | both below stock (2.2x and 4.7x, computed) |
| `sc9_si1000_p1e-3_z_loss1e-2`, rotated surface code d = 9 (801 x 13,969) | 1e-2 | 1.72e-4 (103 / 600k) [1.40e-4, 2.08e-4] | 3.6e-5 (72 / 2M) [2.82e-5, 4.53e-5] | 3.9e-5 (78 / 2M) [3.08e-5, 4.87e-5] | `a05ms31` level with stock; **`a03125c3` 4.4x worse than stock** |
| `sc5_si1000_p1e-3_z_loss1e-2`, rotated surface code d = 5 (145 x 2,003) | 1e-2 | 1.13e-3 (113 / 100k) [9.31e-4, 1.36e-3] | 6.25e-4 (125 / 200k) [5.20e-4, 7.45e-4] | 1.02e-3 (102 / 100k) [8.32e-4, 1.24e-3] | `a03125c3` level with stock; `a05ms31` below |
| `c832_nacal_r2_z_loss2e-2` (26 x 94) | 2e-2 | 2.90e-2 [2.79e-2, 3.00e-2] | 2.77e-2 [2.67e-2, 2.87e-2] | 3.00e-2 [2.89e-2, 3.11e-2] | 100k shots each; `a03125c3` level with stock, `a05ms31` slightly below (intervals disjoint) |
| `c422_nacal_r2_z_loss2e-2` (7 x 23) | 2e-2 | 0.1086 [0.1067, 0.1106] | 0.1079 [0.1060, 0.1099] | 0.1077 [0.1057, 0.1096] | 100k shots each; level |
| `c16_4_4_nacal_r4_z_loss2e-2` (46 x 256) | 2e-2 | 0.2181 [0.2156, 0.2207] | 0.1893 [0.1869, 0.1917] | 0.1839 [0.1815, 0.1863] | 100k shots each; both presets above stock (intervals disjoint), `a03125c3` most |

Readings: the low-cap default is not safe on loss problems in general: on the d = 9 surface code it is 4.4x worse than
stock while `a05ms31` matches stock, and on the small 16-qubit code both presets are above stock; on bb72 both presets
beat stock. As everywhere in this section, measure the preset against stock on the deployment's own loss problem
before accepting it.

**Using the herald information** (the same record; ldpc BP+OSD-0 at the `a03125c3` settings, to which eldpc is bit-identical
at the same config, section 0.1; surface code d = 3, 3 rounds, p = 1e-3, final heralded erasure 0.1, 100,000
shots, the same shots for the four modes):

| mode | LER [95 % CI] |
|---|---|
| blind: the DEM without the herald information | 4.06e-2 [3.93e-2, 4.18e-2] |
| full: herald detectors in the DEM, static priors (what the package decodes today) | 7.09e-3 [6.58e-3, 7.63e-3] |
| per-shot priors from the fired heralds, non-herald rows only | 6.09e-3 [5.62e-3, 6.59e-3] |
| per-shot priors, all rows | 6.07e-3 [5.60e-3, 6.57e-3] |

Exact McNemar tests on the same shots: blind against each other mode p ~ 0; full against per-shot priors p 1.05e-5
(non-herald rows) and 3.55e-6 (all rows); the two prior modes are not different (p 0.868). So the herald information is
worth about 6.7x over blind decoding at this point (computed), and per-shot priors beat the static full DEM by about
1.16x (computed). The package ships no per-shot prior workflow (`docs/SUPPORT.md`, "No loss-aware workflow");
`Decoder.set_priors` is the building block, and its per-call cost on the deployment machine is not measured.

---

## 6. Dead ends

Each row is an idea that was measured as no gain, a loss, or inexact, with the number that decided it. The conditions
are the machine and build of the row (machine key where the record names it; "–" where it does not; all are
development machines and older builds unless stated). A dead end does not automatically transfer to a different
machine (section 6.5). Do not retry one without a new argument and a measurement. Where the registry records two
independent measurements, both are given.

### 6.1 Accuracy and message formats

| idea | measured result | machine | source |
|---|---|---|---|
| min-LLR OSD-0 column ordering (columns ordered by each column's running minimum posterior over the iterations, as in CUDA-Q's `osd_init_method="min_llr"`) | worse wherever countable: bb144 xyz p = 3e-3 a03125c3 1.31x, a0375c7 1.13x; p = 5e-3 a03125c3 1.14x, a0375c7 1.19x (100k fresh shots per cell, reproduced independently on independent shots; McNemar p 3e-3 to 4e-181); p = 1e-3 a03125c3 partial 2.12M shots: 35 failures with min-LLR vs 23 without; cost +3.3 % p50 (~+1.25 µs) and 6 of 28 kernels grow stack or spill. An earlier "2.5x worse" figure came from an implementation defect and is withdrawn. The effect grows with the iteration count: at 31 iterations min-LLR gave ~1.5x fewer failures (not significant), at 60 iterations (alpha 0.625, p = 3e-3) ~10x fewer (213 -> 21 failures of 2,000 shots), a regime where BP+OSD is itself poor (failure rate 0.107 against 0.0096 for a03125c3) and min-LLR only brings it back to a03125c3's level; so the direction is closed at <= 31 iterations, not universally. Not in the shipped core | B, CUDA 13.3 | development record: min-LLR study; development record: min-LLR iteration follow-up |
| 2-cycle early exit as an exact change | not bit-exact (posteriors drift); alpha 0.5 BP on bb144 is 94.6 % aperiodic, so little to gain | – | `tuning/data/knob_facts.json: early_exit_2cycle` |
| f32 / i16 / i8 messages as a free speed-up | every format changes outputs, so none is shippable; accuracy at the tuned point f32 1.00, i16 1.0 [0.928, 1.08], i8 0.984 [0.858, 1.13] (10M fresh shots); f32's per-iteration gain (9.08 -> 4.39 µs on bb288, 5090) did not make the library faster end to end | RTX 5090 (host not recorded) | `tuning/data/knob_facts.json: msg_format` |
| even caps at alpha 0.5; caps 1-2 | section 3.1 | – | `tuning/data/knob_facts.json: cap`; `tuning/data/knob_facts.json: alpha` |
| "convergence by flips" (detect convergence from bit flips) | +98.8 / +109 µs per shot (+13-15 %), measured twice, independently; a second machine +75.5 / +77.9 µs p50; a third implementation +141.9 / +139.1 µs p50 | D; B; G | `tuning/data/knob_facts.json: dead_ends` ("convergence by flips", three entries) |
| publish the converged verdict with a third grid barrier per iteration | +48.7 µs per shot against 0.15 µs saved | priced from RTX 5090 probe values | `tuning/data/knob_facts.json: dead_ends` |

### 6.2 OSD structure and host walk

| idea | measured result | machine | source |
|---|---|---|---|
| one full stable LSD radix sort instead of the batch machinery | 59-107x the walk (3.8-3.9 ms vs 36-66 µs on bb144 xyz) | G | `tuning/data/knob_facts.json: dead_ends` |
| rank-3 first batch; sorting the whole batch at the first `next()`; one `nth_element` at kPrefix | +5.0..+5.1 µs per shot; +3.0..+3.9 µs; +2.6 µs p50 | G | `tuning/data/knob_facts.json: dead_ends` |
| n-scaled stride sample (n/64, rank 6) | walk +6.3..+6.6 µs bb288 z, +3.6 bb144 z, +16.9..+17.5 bb144 xyz a05ms31 | G | `tuning/data/knob_facts.json: dead_ends` |
| looser first threshold via a rank target | init +8.7..+11.3 µs | G | `tuning/data/knob_facts.json: dead_ends` |
| `kOsdS` 512 -> 256 | +0.8 µs p50 / +6.5 µs p99; repicks 8 -> 22 of 2,000 | 84-SM RTX 5080 machines | `tuning/data/knob_facts.json: dead_ends`; `tuning/data/knob_facts.json: osd_cap_rungs` |
| larger rung one (384 / 640 -> 512 / 896) | +0.8 µs p50 in both create orders, although p99.9 -8.9 and max -11.7 µs; re-price only with a sort linear in the set size | 84-SM RTX 5080 machines | `tuning/data/knob_facts.json: osd_cap_rungs` |
| bit-packed O with AND + POPCNT observables | 63.9 vs 21.6 µs per shot (2.96x slower) | G | `tuning/data/knob_facts.json: dead_ends` |
| observables block width 16 or 64 instead of 32 | +0.67 / +1.40 µs (OSD class), +1.04 / +2.08 µs (converged class) | G | `tuning/data/knob_facts.json: dead_ends` |
| clear only the previous support instead of `memset(n)` of the correction | +1.30 / +3.49 µs and +2.01 / +1.78 µs, measured twice, independently; a variant that was faster (-1.57..-1.76 µs) is not allowed: the correction buffer is caller-owned with undefined contents | G | `tuning/data/knob_facts.json: dead_ends` |
| derive the hard decision on the host from the posterior | host pass 65.8-65.9 µs: net +62..+67 µs | G | `tuning/data/knob_facts.json: dead_ends` |
| device prefix selection as a standalone pass; histogram + compaction + 16 KB D2H | 38.2-165 µs (four forms) vs <= 20-31 µs of D2H saved (5070 Ti); 49.1-50.2 vs 25.5-25.8 µs for the full 608 KB D2H (RTX 5080) | G; B | `tuning/data/knob_facts.json: dead_ends` |
| Four-Russians / bit-sliced elimination; reduced-echelon (Gauss-Jordan) basis; column-block elimination | pivot vectors average 3.7 bits, word work <= 10-20 %; 1.51-2.61x slower; 1.04-1.54x slower | – | `tuning/data/knob_facts.json: dead_ends` |
| AVX-512 gathers in the walk | 1.075-1.09x slower (vpgatherdd is microcoded on Zen 5) | – | `tuning/data/knob_facts.json: dead_ends` |
| dense micro-optimisation bundle of the elimination | elimination p50 +0.01 / +0.27 µs against an A/A floor of 0.15-0.32 µs | G | `tuning/data/knob_facts.json: dead_ends` |
| column-domain `H x != s` re-check on the vendored path | 235 -> 3.5 µs but unreachable: a DEM-sampled syndrome is in the image of H (0 of 3,600 decodes deferred) | G | `tuning/data/knob_facts.json: dead_ends` |

### 6.3 BP kernel

| idea | measured result | machine | source |
|---|---|---|---|
| hand-written grid barrier | 1.49x the tag-checked cg barrier, 2.06-2.09x bare; the conformant form 1,062.8 ns = 1.88x cg; a 434 ns relaxed form is not conformant | J; D | `tuning/data/knob_facts.json: dead_ends` |
| fire-and-forget grid barrier (release arrival + acquire poll) | cap 3 p50 +0.0 / p99 +0.4 µs, a05ms31 +1.4 µs, two-point slope ratio 1.001 | B | `tuning/data/knob_facts.json: dead_ends` |
| one kernel launch per pass instead of grid barriers | 3.1-3.5 µs per launch = 4.3-4.8x the barrier | J | `tuning/data/knob_facts.json: dead_ends` |
| partial / dependency-tracked grid barrier | median reach 48 / 50 / 70 of 84 blocks | J | `tuning/data/knob_facts.json: dead_ends` |
| fewer blocks / grid shaping | barrier flat at 0.513-0.574 µs over 16-168 blocks; bb144 xyz 843.7 µs at 84 blocks vs 1,824.8 at 21 | J; D | `tuning/data/knob_facts.json: dead_ends` |
| cluster barriers: 16-block cluster; hierarchical cluster barrier; clusters of 7 / 8 at 84 SMs | 7-12x the grid barrier; 550-1,076 ns vs 540-545 ns for the grid barrier (ties at best); do not place (7 GPCs x 12 SMs) | –; RTX 5080 | `tuning/data/knob_facts.json: dead_ends` |
| more warps (252 x 256 instead of 84 x 512); other shapes; two blocks per SM; 768 / 1,024 threads per SM | 20.76 vs 18.37 µs per iteration (+13 %); 384 threads 1.12x, 256 threads 1.29x, 42 blocks 1.55x; iterations +31..+65 % (64 registers + spills); stack 176 / 240 B | D; D; –; D | `tuning/data/knob_facts.json: dead_ends` |
| two columns per thread | two-point slope ratio 1.263, +10.2 µs | – | `tuning/data/knob_facts.json: dead_ends` |
| min-only reduction for the second minimum | inexact: the second minimum differs in 1,518 of 1,872 rows | D | `tuning/data/knob_facts.json: dead_ends` |
| four rows per warp, 8 lanes per row | +1.83 / +1.97 µs per iteration (+18..+20 %), measured twice, independently | G | `tuning/data/knob_facts.json: dead_ends` |
| row-ordered (CSR) message copy | check pass -4.38 µs per iteration, column pass +5.21..+5.27: net +0.894 | G | `tuning/data/knob_facts.json: dead_ends` |
| ILV on high-waste in-place problems without the 12 % gate | hgp98 z +11.1 % and +11.2..+12.0 % (measured twice, independently); bb144 z +3..+8 % | G | `tuning/data/knob_facts.json: dead_ends` |
| edge-balanced (LPT) column partition; degree-sort indirection on uniform-degree problems; a random per-block permutation | two-point slope ratio 1.178-1.247 (rows per block 175-225 -> 1,770-1,810: locality lost); bb288 z +1.99 / +1.71 %; slower by 45.45 / 53.36 µs per shot | G; G; J | `tuning/data/knob_facts.json: dead_ends` |
| partial b2c window selected per edge | two-point slope ratio 0.9615 vs 0.9230 for the full window | – | `tuning/data/knob_facts.json: dead_ends` |
| last-toucher row push | two-point slope ratio 1.3316 | J | `tuning/data/knob_facts.json: dead_ends` |
| fold the exit read into the barrier; `__ldcg` hints; one XOR reduction for both parities; `kDualReg = false`; `__restrict__`; `#pragma unroll 3`; u16 row index in global memory | +160.7 ns per iteration; +0.007 µs; +0.010 / +0.005 µs; inside the floor; inside A/A; +2.9..+3.1 %; +2.32 % column pass | D | `tuning/data/knob_facts.json: dead_ends` |
| remove the trailing mismatch pass (in-place problems) | -0.08 / -0.72 µs per shot, inside the probe floor | B | `tuning/data/knob_facts.json: dead_ends` |
| more blocks per SM on a 70-SM part | 210 x 256: 29.4-29.7 vs 70 x 512: 27.5-27.6 µs per iteration | G | `tuning/data/knob_facts.json: dead_ends` |

### 6.4 Launch and hand-off

| idea | measured result | machine | source |
|---|---|---|---|
| persistent kernel | 0.4-2 µs only on an early build; later solo a wash (87.99 vs 87.47 µs); upper bound of the prize ~2.1 µs (a 5.35 µs launch replaced by a 3.24 µs flag round trip, F probes) | – | `tuning/data/knob_facts.json: dead_ends`; development record: design record |
| CUDA graph per shot | ~1.4 µs in-process, whole call inside the floor; it allocates on the decode path | B | `tuning/data/knob_facts.json: dead_ends` |
| static CUDA graph + a mapped shot block | launch -0.38 µs, device +0.77 µs (block 0's PCIe read of the shot block): net 0 | – | `tuning/data/knob_facts.json: dead_ends` |
| `cudaLaunchKernelEx`, a plain instead of cooperative launch, the driver-API launch | 0.02-0.04 µs; 0.02 µs; -0.10 µs | B; – | `tuning/data/knob_facts.json: dead_ends` |
| event-query spin instead of the stream sync | 0.17-1.04 µs (sub-µs); +0.9 µs behind work | B; G | `tuning/data/knob_facts.json: dead_ends` |
| launch-ring stall levers: smaller parameter block, dummy launches at the predicted wrap, event records or memsets as absorbers, `CUDA_DEVICE_MAX_CONNECTIONS` | no lever with the real decoder on drivers 595 / 615 (stall share 5.5 -> 4.5 % with p99 unchanged, or no change) | RTX 5080 machines | `tuning/data/knob_facts.json: dead_ends` (launch-tail entries); `tuning/data/knob_facts.json: iommu_passthrough` (notes) |
| byte-at-a-time host bit-pack of the syndrome | 2.995 µs vs 0.611-0.631 µs | G | `tuning/data/knob_facts.json: dead_ends` |
| one contiguous D2H for [decisions, status] instead of two copies | 0.10-0.20 µs: the round trip is the sync, not the copy | G | `tuning/data/knob_facts.json: dead_ends` |
| pageable host staging for the posterior; chunked posterior D2H to start OSD early | 608 KB D2H 68-82 µs vs 26-30 µs pinned (2.6-2.8x); first chunk 2.0-2.2 µs earlier but all chunks 39.2 vs 26.4 µs | G | `tuning/data/knob_facts.json: dead_ends` |
| cross-shot pipelined batch decode | throughput only (256-shot batch period 52.5 -> ~43.5 µs); one-call latency unchanged; not shipped | – | `tuning/data/knob_facts.json: dead_ends` |
| not shipped, not dead: mapped decisions on the converged path | bb144 z converged class -3.8 µs at every iteration count, but bb288 z a05ms31 OSD class +2.2 µs (+0.8 %) and long-cap xyz +0.3 %; needs a per-class gate and a failure test on its paths | D, CUDA 13.0 | `tuning/data/knob_facts.json: mapped_decisions` |
| not shipped, not dead: loop hoist in the BP loop | -0.4 µs p50 / -1.1 µs p99, two-point slope ratio 0.964; stack growth on the non-row-slot kernels | D, CUDA 13.0 | `tuning/data/knob_facts.json: loop_hoist` |
| not shipped: two-core look-ahead host walk | estimate only: deep-walk p99 -1.4..-2.5 µs; needs a library-owned thread | – | `tuning/data/knob_facts.json: dead_ends` |

### 6.5 Platform-scoped and keep-warm dead ends

| idea | measured result | machine | source |
|---|---|---|---|
| zero-copy hand-off (kernel writes the 8n posterior into mapped host memory) | 237.3 / 159.5 µs for 608 KB vs 26-37 µs pinned D2H (4.3-7.6x slower) on RTX 5070 Ti + Zen 2; **not slower on RTX 5080 + Zen 5** (28.2 vs 30.8 µs for fill + D2H) | G; H | `tuning/data/knob_facts.json: dead_ends` |
| keep-warm host-table sweep flag (value 2) | removed: placed right it gained nothing, alone it evicted more than it warmed; the shipped core rejects the flag | – | `src/core/fastbpgpu6g.cpp` (comment); `src/core/fastbp_ext.h` |
| coarse host warm-up sweep (~20 MB) alone | 47.57 / 61.71 / 95.59 µs vs control 46.84 / 63.14 / 95.34 (p50 +0.3..+0.6, evicts working lines) | A-oc, build 6e | `tuning/data/knob_facts.json: keep_warm_call` |
| keepalive every 2 ms | clock only ~2,915 MHz: 46.64 / 62.75 / 95.08 µs, no gain | A-oc, build 6e | `tuning/data/knob_facts.json: keep_warm_call` |
| keep-warm inside the library with no caller call | the library starts no thread; revive only if a caller-owned idle hook ships | – | `tuning/data/knob_facts.json: dead_ends` |
| forcing the PCIe link generation | the GPU leaves P1 after 1.3-2.3 s idle and reaches P8 after 3.8-4.8 s, so no decode stream at these cadences sees P8 or a lower link speed | A-oc | `tuning/data/knob_facts.json: call_cadence` |
| keep-warm in a back-to-back loop | the call after a keep-warm against the call after none: p50 1.0046 (bb288 z) and 0.9996 (bb144 xyz), p99 1.1267 on bb288 z (the empty kernel sits ahead of the next decode) | M | `tools/results/host-xeon/host-xeon.json` (section 4.4) |
| `-march=sapphirerapids`, `-march=x86-64-v4` or clang++ 16 for the host core | no difference on 10 of 10 p50 / p99 comparisons for each `-march` build; clang slower on 1 of 6, no difference on 5 | M | `tools/results/host-xeon/host-xeon.json` (section 4.9) |
| a preset-keyed (cap-aware) core-variant rule on Ada | no cap threshold is a clean win on every measured problem and preset on both boards: bb144 xyz wants the default build at `a05c11` and `ms30`, bb288 z (and hgp98 xyz at `ms30`) want `ada64` | K; L | `tools/results/ada-sm-budget/full-bench.json` (K); `tools/results/ada-sm-budget/full-bench.json` (L) (section 4.6) |
| a budget-56 Ada core variant | within noise of budget 64 on the RTX 6000 Ada (direct full-bench pair: no difference at p50 on bb144 xyz and bb144 z; 64 better on bb288 z and at p99.9); 64 favoured on the RTX 4090 (56 / 64 geometric mean 1.0204) | K; L | `tools/results/ada-sm-budget/full-bench.json` (K); `tools/results/ada-sm-budget/full-bench.json` (L) (section 4.6) |
| CPU core across two CCDs (10, 12 or 16 threads) or on SMT second contexts | 2.11-2.41x the p50 of 8 threads on one CCD; SMT second contexts 1.001-1.017x (no gain) | E | development record: CPU-core study; `tools/results/cpu/placement.json` (section 4.11) |

The two clear cases of conditions mattering are the zero-copy posterior (bad on one GPU/host pair, not on another) and
the idle tail (host-side on F, GPU clock on A). If your machine, driver or toolkit differs from a row's, re-measure
before spending time on it.

### 6.6 Measurement practice that misled

| practice | what went wrong | source |
|---|---|---|
| two-point slope with one process per create order | median -1.51 %, process-ratio spread 2.54 % peak-to-peak vs 0.65 % with three | `tuning/data/knob_facts.json: dead_ends` |
| deciding BP-loop steps on the two-point slope alone | ~5 µs of post-BP damage was invisible to it; decide on the whole call | `tuning/data/knob_facts.json: dead_ends` |
| offline micro-benchmarks that repeat one input; warm-replay host benchmarks | an operation measured 1.2 µs offline cost 5.6 µs in situ; warm replays over-trained the branch predictor | `tuning/data/knob_facts.json: dead_ends` |
| aiming a change at one session's p99 term | a single-session outlier misdirected two attempts; take the stage breakdown twice | `tuning/data/knob_facts.json: dead_ends` |
| timing steps below ~3 % on shared machines or across sessions / GPUs of one machine | the same code measured 13 µs apart; one GPU of a two-GPU machine was ~4 % slower; A/A floors p50 0.048-0.21, p99 0.17-4.2 under neighbour load | `tuning/data/knob_facts.json: dead_ends` |
| deciding kernel steps on CUDA 13.0 rows | one step spilled 40 B on 13.0 and 8 B on 13.3 | `tuning/data/knob_facts.json: dead_ends` |
| the library's sha256 as provenance | nvcc embeds a random temporary-file token, so rebuilds of the same source differ; compare `build_info` and normalised code instead | `tuning/data/knob_facts.json: dead_ends` |
| judging a bimodal preset on its pooled p50 | the pooled p50 falls on the cliff between classes | `tuning/data/knob_facts.json: dead_ends` |
| failure-injection tests that pass when nothing fired | three of four classes never fired; check that each class fires | `tuning/data/knob_facts.json: dead_ends` |
| reading a paired ratio of two launch plans as their solo ratio | both handles share one process, and a 64-block build made the default side faster than against itself: paired 0.7409 / 0.6604 against solo (one library per process) 0.6567 / 0.5877 on bb144 xyz / bb288 z (L); paired ratios of such builds are conservative, and two close builds must be ordered by a direct pair | `tools/results/ada-sm-budget/full-bench.json` (L) (section 4.6) |
| comparing the two sides of a pair by call position or creation order | the first call of each interleaved pair was slower at p50 in every run (first / second called 1.032 on bb144 xyz, 1.059 on bb288 z; 1.014-1.047 with per-call buffer clearing or a 0 / 1 ms gap), and on bb144 xyz the first-created handle was slower too (1.012, 11 of 12 runs) (M); the seeded interleave and the alternating create order average both out (section 7.1) | `tools/results/host-xeon/host-xeon.json` |
| a host-side A/B on a launch plan other than the one that will run | the AVX2 penalty on bb144 xyz was p50 1.0363 with the 64-block plan and 1.0193 with the 142-block plan: a longer GPU call dilutes host terms; measure host effects with the core variant the deployment uses | `tools/results/host-xeon/host-xeon.json` (section 4.9) |

---

## 7. Statistical method

Reproduce this method when you measure, or the numbers are not comparable with the tables above. Sources: development
record: protocol, unless a row says otherwise.

### 7.1 Latency

| element | rule |
|---|---|
| paired, interleaved shots | candidate and reference decode the same shots in the same process; the per-shot order alternates by a seeded pseudo-random interleave; the timed region is one decode call and nothing else |
| alternating create order | processes alternate candidate-first and reference-first handle creation (allocation order sets memory placement); report both orders and the pooled value |
| replication | at least 3 main processes and at least 3 A/A processes (the reference timed against itself) per comparison; identical decoders differ by a per-process constant (+-0.7 % sd on bb144, up to +-4.7 % on small codes, 10-20x one run's CI half-width) |
| the A/A floor | `floor = max(min_effect, max over A/A runs of (abs(r_i - 1) + h_i))`, `min_effect` 0.02; "faster" only if every main run's upper CI bound is below 1 - floor, "slower" only if every lower bound is above 1 + floor; otherwise no difference |
| quantiles | nearest-rank p50 / p99 / p99.9; per-shot interleaved runs for ratios, solo (single-handle) runs for absolute latency budgets |
| classes | report converged and OSD-path classes separately whenever the preset is bimodal (section 3.2) |
| environment | pinned to an isolated core, quiet machine (neighbour-busy fraction <= 5 % for absolute rows), call cadence and keep-warm state stated; the timer's overhead is measured and never subtracted |
| two-point slope | per-iteration estimator `(T(cap 31) - T(cap 1)) / 30` in one session, >= 3 processes per create order |

A/A bands in the records:

| machine / session | A/A band | source |
|---|---|---|
| A-oc, final bench, a03125c3 | p50 0.16 µs, p99 0.22 µs (0.5 % of the call) | `tools/results/final-bench/final-bench.json` |
| A-stock, final bench, a03125c3 | p50 0.12 µs, p99 0.18 µs | `tools/results/final-bench/final-bench.json` |
| A-oc, final bench, a05ms31 | p50 0.49 µs, p99 0.47 µs | `tools/results/final-bench/final-bench.json` |
| B, final build | A/A median p50 0.9996 / floor 2 %; p99 1.0053 / floor 7.6 % | development record: design record |
| a shared development machine, whole call | p50 0.048-0.21, p99 0.17-4.2 (floors, neighbour load) | `tuning/data/knob_facts.json: dead_ends` |

On machine A a 0.3 µs change at 32 µs is readable (its final-bench A/A band above); on a shared machine a 2 % change
often is not (the knob-facts floors).

p99 budgets are host-dependent: measured on an RTX 6000 Ada with a two-socket EPYC 7542 host (Zen 2), 1.0-1.2 % of Python
`decode_full` calls carried an extra 65-99 µs host tail (median about 80 µs) at every configuration, in the converged and
the OSD class alike, so the p99 sat at about p50 + 75 µs (development record: tuner re-validation); tune on the deployment
machine.

### 7.2 Accuracy (LER)

| element | rule | source |
|---|---|---|
| failure | a shot fails if any predicted observable differs from the truth; an abstaining shot is discarded in the descriptive LER and counts as a failure in the paired test | development record: protocol |
| pairing | candidate and reference on the same shots; report both failure counts, the ratio, and the discordant counts with an exact McNemar test | development record: protocol |
| paired bound | one-sided 95 % bootstrap upper bound on `F_cand / F_ref` (2,000 resamples); pass iff upper <= 1 + delta, delta 0.05; at least 100 reference failures | development record: protocol |
| absolute bound | one-sided 99 % Clopper-Pearson upper bound on the LER against the bar (3M-shot confirmation: CP99 < 1e-4) | `tuning/data/knob_facts.json: alpha` |
| descriptive intervals | 95 % Wilson; they describe, they never decide | development record: protocol |
| per-round LER | `1 - (1 - LER)^(1 / rounds)` | development record: protocol |
| fresh seeds | each stage (screen, confirmation) draws its own shots from a fresh seed; never reuse a screen's shots to confirm it | development record: protocol |
| stage sizes and stopping | the records use pre-declared stage sizes (a 2M-shot screen, a pre-registered 10M confirmation, a 3M + 500k confirmation) and no sequential stopping rule; declare the stage size and the bound before running, and do not stop early on a favourable look | `tuning/data/knob_facts.json: alpha` |
| sizing | for "clearly below the bar" the number of shots follows from the bar and the expected LER (CP99 of k failures in N shots below the bar); a tight ratio needs many more shots | section 5.1 |

### 7.3 Reporting checklist for a new tuning result

| item | required with the number |
|---|---|
| machine | GPU model, SM count, L2, driver, clocks (stock or the exact overclock profile), power limit; CPU model and ISA core loaded (`eldpc.isa()`); bench conditions (isolated cores, pinned core) |
| build | `eldpc.build_info()` (compiler, flags, CUDA version, architectures, budgets) and `nvcc --version`; not the library hash |
| cadence | the gap between calls and the keep-warm schedule |
| runs | processes per create order, A/A processes, the A/A floor, both orders and the pooled value |
| shots | the shot-set identity (hash or seed rule) and the number of timed shots |
| latency | p50 / p99 / p99.9 per class (converged, OSD path) and pooled, single-shot C call (Python overhead stated separately) |
| LER | noise point, shots, failures, reference failures, paired ratio and its one-sided bound, McNemar p, absolute LER and its CP99, stock's LER at the same p |
| exactness | after any build or toolkit change: the package bit-identity tests (`tests/test_bit_identity.py`, `tests/test_isa.py`) plus an identity run on whole shot sets of every tuned problem and shipped config on the deployment GPU; tool, shots and result, or "not checked" |
| problem | m, n, nnz(H), max row degree, k, rounds, basis, and the create-time shape if a probe build was used |

---

## 8. Known gaps

- The primary deployment class (RTX 6000 Ada, sm_89, 142 SMs, 96 MiB L2; Sapphire Rapids Xeon W host) is measured
  in halves: the RTX 6000 Ada's device-side budget ratios, long-cap rows and create-time shapes are measured on machine
  K (section 4.6), whose Zen 2 host is AVX2-only, so there is no deployment latency, no paired keep-warm A/B on the
  board and no host term on the deployment host. The Sapphire Rapids-W host family is measured on machine N with an
  L40S, not with the RTX 6000 Ada board (whole-call latency, host ISA and core variant A/B; sections 0.2 and 4.9);
  pinning, SMT and NUMA placement are measured only on machine M's Emerald Rapids Xeon Silver with an L40S (sections
  4.4, 4.9, 4.10). Every latency must be re-measured on the deployment machine.
- No stock LER at p = 2e-3 or 3e-3: the ratios there are against the (0.5, 31) reference.
- `a04375c3` has no bb144 xyz latency row and no bb144 z LER above p = 1e-3; `ms100` and `ms30a075` have no tuning
  records; `ms30` has no final-build latency row.
- The p = 2e-3 and 3e-3 points for a03125c3 on re-noised variants are single samples (1M and 500k shots).
- The min-LLR p = 1e-3 cell is partial (2.12M shots).
- The `-march` builds and the pinning rows are from one shared Emerald Rapids host (M); the whole-call AVX2 vs AVX-512
  rows are from M and from the Sapphire Rapids-W host N (section 4.9); no CPU-core row on a Xeon or a two-socket host
  (section 4.11).
- Locked graphics clocks and IOMMU passthrough are untested; the GDRCopy-style path is an estimate and is not shipped.
- The shipped library cannot report its create-time shape; a probe build is needed (section 2.5).
- The Python per-call overhead is not in the records; `tools/measure_decode_overhead.py` measures it.
- The registry predates the final build: its `components` fit and `target_points` are builds 6e / 6c; its keep-warm
  entries describe the prototype and an unshipped flag (section 2.3).
- Joint against separate X/Z decoding (section 3.5) is measured on bb144 xyz at p = 1e-3 and 3e-3 and on hgp98 xyz at
  1e-3, two presets plus two crossed configs; the joint advantage at `a03125c3`, p = 1e-3, rests on one 5M-shot run whose
  1M-shot fresh-seed confirmation tied. No latency of the separate route is recorded.
- Loss (section 5.6): final-measurement heralded erasure only; LERs on six problems with no paired test; the per-shot
  prior workflow is measured only on a d = 3 surface code with ldpc, and its `set_priors` cost is not measured.

---

## Sources

Three kinds of source are cited.

1. **Shipped files** of this package, by path: `src/core/fastbp_ext.h`, `src/core/fastbpgpu6g.cpp`,
   `src/core/bp_gpu.cu`, `src/core/bp_engine.hpp`, `src/core/osd0_engine.hpp`, `src/core/osd_caps.hpp`,
   `src/port/eldpc_front.cpp`, `include/eldpc.h`, `python/eldpc/__init__.py`, `python/eldpc/_config.py`,
   `python/eldpc/_abi.py`, `python/eldpc/_dem.py`, `python/eldpc/_devices.py`, `python/eldpc/_variant.py`,
   `src/port/core_variant.h`, `CMakeLists.txt`, `tests/test_bit_identity.py`, `tests/test_isa.py`,
   `tools/measure_decode_overhead.py`, `verify/eldpc_verify/circuits.py`, `tests/verify/test_herald.py`; and the
   shipped result summaries, each directory with a README that says how it was measured (summary statistics and
   per-row aggregates of the measurement records, numbers as recorded):
   - `tools/results/final-bench/final-bench.json`: the pre-registered latency bench of the final build on machine A
     at both clock profiles: 221 runs per profile, 2.6M paired shots per profile, 0 output mismatches, 0 Xid;
     single-handle cap curve, presets, z problems, paired rows against builds 6e and 6f, cadence and keep-warm
     conditions;
   - `tools/results/stock-baseline/stock-baseline.json`: the stock-baseline measurements on machine A's host (ldpc
     2.4.1, one core) and the LERs of the stock comparisons, with the pooled fresh-shot LERs of a03125c3 (bb144 xyz)
     and a04375c3 (bb144 z);
   - `tools/results/ada-sm-budget/rtx6000ada-142sm.json`, `rtx4090-128sm.json` and `full-bench.json`: the
     compile-time SM-budget builds of the final core on machines K (budgets 40-80) and L (48-72): exactness gates,
     paired A/A screens, full bench, direct pairs, layout probes, stage probes, solo and companion-handle checks, the
     long-cap follow-up on K, and machine M's earlier launch-geometry screen;
   - `tools/results/host-xeon/host-xeon.json`: host ISA builds, pinning conditions, keep-warm and cadence of the final
     core with the 64-block plan on machine M, with a re-measurement of its main rows (the corrected values are the
     ones quoted);
   - `tools/results/joint-vs-separate/joint-vs-separate.json`: LER of joint xyz decoding against separate merged X/Z
     sector decoding on machine E (accuracy only);
   - `tools/results/heralded-loss/heralded-loss.json`: the corrected herald insertion, LER on six loss problems
     (machine E) and the d = 3 herald-use comparison with ldpc (accuracy only);
   - `tools/results/cpu/*.json` (the CPU core's records).
2. **The knob facts**, `tuning/data/knob_facts.json: <key>`: the records of the package's tuning measurements, with
   labels and wording generalised and numbers as recorded (its `about` defines every field, machine, build and study
   label). A key is a top-level group (`target`, `ler_reference`, `components`, `components_legacy`, `target_points`,
   `bench`, `dead_ends`) or a key of its `knobs` map (`alpha`, `cap`, `gpu_oc`, ...); a name in parentheses after a key
   is a sub-group or a record's `metric`; `dead_ends` records are matched by their `knob` text. The registry predates
   the final build.
3. **Development records** (not shipped), by short name. They are not part of the package: a number whose only
   source is one of them can be traced by that name but not re-checked from the package.
   - *design record*: the decoder's design and measurement record: probes, stage breakdowns, sizing rules, budget
     builds (machines B, C, F).
   - *validation runs*: runs on machine E (RTX 5080 + Ryzen 9 7950X3D, not timing-grade): the keep-warm identity
     selftest and the paired LER runs at p = 2e-3 and 3e-3 on re-noised variants, with their shot-set hashes and
     seeds. No timing from machine E is used here.
   - *min-LLR study*: the min-LLR OSD-0 ordering experiment on machine B, with its independent reproduction.
   - *min-LLR iteration follow-up*: the min-LLR ordering at 3, 31 and 60 BP iterations on bb144 xyz (fresh shots,
     pooled over three independent samples).
   - *protocol*: the benchmark protocol specification (pairing, interleaving, A/A floors, bootstrap and McNemar tests,
     seeds).
   - *parity record*: bit identity of the decoder against the ldpc reference implementation.
   - *CPU-core study*: exactness, screens and same-session A/B rows of the CPU core on machine E.
   - *tuner re-validation*: tuner and benchmark runs on an RTX 6000 Ada with a two-socket EPYC 7542 host: the per-call
     latency-tail classes, the tuner's screens, latency runs and confirmations, and fresh 1M-shot LERs of the chosen
     configurations.
