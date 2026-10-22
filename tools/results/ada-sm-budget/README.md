# SM-budget records on the Ada boards

Compile-time SM budget (`DBENCH_GPU_SM_BUDGET`) builds of the frozen decoder core measured against the default build
in the reference harness on two AD102 boards. The shipped `ada64` core variant is the budget-64 build. Machines (keys
of tuning/KNOWLEDGE.md section 0.2):

- **K**: RTX 6000 Ada Generation (compute capability 8.9, 142 SMs, 96 MiB L2, 300 W power limit), 2x EPYC 7542
  (Zen 2, AVX2 only: harness builds with the AVX2 host flags), bare metal (no hypervisor flag) in a container, driver
  580.126.09, CUDA 13.3.73, SM clock not locked; every timed process pinned to CPU 0 on the GPU's NUMA node.
- **L**: RTX 4090 (compute capability 8.9, 128 SMs, 72 MiB L2, 400 W power limit), Ryzen 9 7900 (Zen 4, AVX-512
  host flags), virtual-machine guest with other tenants, driver 610.43.02, CUDA 13.3.73; every timed process pinned
  to CPU 0.
- **M** (one table only): the earlier L40S screen (142 SMs, driver 595.84, 2x Xeon Silver 4514Y).

## Files

| file | what | design and shots |
|---|---|---|
| `rtx6000ada-142sm.json` | K: per budget (default, 40-80) the exactness flag and the paired screens against the default build (pooled p50 / p99 / p99.9, ratios, verdicts) on five to seven problems (bb144 xyz, bb144 z, bb288 z, bb72 z, hgp98 z; hgp98 xyz and sc9 z for most budgets) at `a03125c3`; budget 56 and 64 also at `a05c11`; the long-cap follow-up rows of budget 64 (`follow_up`) | 3 processes per create order x {main, A/A}, 20,000 shots per side per run: 120,000 shots per side pooled |
| `rtx4090-128sm.json` | L: the same for budgets 48-72, with the long-cap screens of budget 64 (`a05c11`, `ms30`) | the same |
| `full-bench.json` | K and L: the exactness gate (builds x rows, shots per row, mismatches), the screens' output check, the back-to-back full bench of budgets 56 and 64 against the default and its 10 ms cadence rows (without and with keep-warm), the direct pairs of close budgets (full bench on K, screens on both), the pick by geometric mean, the stage probes, the create-time layout probes, the one-library-per-process (solo) checks, the companion-handle analysis (L), the long-cap follow-up with its A/A bands and host load (K), and M's earlier L40S screen as tabulated by the K sweep | full bench: 4 processes per create order x {main, A/A}, 20,000 shots per side per run, back to back: 160,000 shots per side pooled (6 processes, 240,000 per side, for K's direct 56-against-64 pair); cadence rows: 2 processes per create order x 500 shots (2,000 per side), a 10 ms busy-wait gap before each call, keep-warm every 500 us of the gap and 25 us before the call; stage probes: K 4 processes x 1,000 shots, L 64,000 shots per build; solo: 3 rounds x 20,000 shots, one library per process |

Method common to all files: candidate and reference are built from the same six core sources with one extra
`-DDBENCH_GPU_SM_BUDGET=<n>`; before any timing an exactness gate compared correction, observables, status word and
soft output per shot against the reference and the default build (0 mismatches for every build); the paired sessions
decode the same shots in one process with a pseudo-random per-shot interleave, alternate the create order, and pair
every main run with an A/A run (the reference against itself); quantiles are pooled nearest-rank; a verdict is
"faster" or "slower" only when every main run is separated from the A/A band (tuning/KNOWLEDGE.md section 7.1). The
stage probes and layout probes are measurement builds of the same sources that are never shipped. The host of K is
not the deployment class, and L is a virtual-machine guest: the ratios are device-side comparisons; absolute times are
indicative only.

All files are summary statistics extracted from the measurements' results files, numbers as recorded. Units in
`full-bench.json`: `*_ns` pooled quantiles in nanoseconds, `*_us` microseconds as recorded, ratio = candidate /
reference (< 1 = the candidate is faster), budget 0 = the default build.

## Through the package

`package-rtx6000ada/`: the two core variants of the installed package timed through its C API on machine K
(`bench/bench.py c-latency --quick` with `ELDPC_DEVICE_CLASS=ada64` and `=default`, a screen): `ada64` / default
x0.714 at the p50 on bb144 xyz at `a03125c3`; its README has the commands.

## Cited by

- `tuning/KNOWLEDGE.md`: section 0.2 (machines K, L), 2.3 (core variant), 4.4 (the machine K cadence paragraph:
  `full-bench.json` K `full_bench`, cadence rows), 4.6 (the ratio table: the two summaries; method counts, budget 56
  against 64, the full bench, long caps, stage probes, create-time shapes, the companion-handle rows: `full-bench.json`),
  6.5 and 6.6 (the core-variant, budget-56 and paired-ratio rows), 8.
- `docs/SUPPORT.md`: "GPU architectures", "Host policy" (core variant on Ada), "Known limitations".
- `docs/BUILD.md`: "Core variants".
- `docs/CLAIMS.md`: H7 (the harness ratios and, through the package, `package-rtx6000ada/`).
- `README.md`: "Long iteration caps on Ada".
