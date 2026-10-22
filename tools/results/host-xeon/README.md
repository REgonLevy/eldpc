# Host study on a two-socket Xeon host (machine M)

`host-xeon.json` is an extract of the host study of the frozen GPU core on machine M of
[tuning/KNOWLEDGE.md](../../../tuning/KNOWLEDGE.md) section 0.2: an NVIDIA L40S (compute
capability 8.9, 142 SMs, 350 W power limit, ECC on, driver 595.84) in a host with 2x Xeon Silver
4514Y (Emerald Rapids, AVX-512; 16 cores x 2 threads per socket, two NUMA nodes), bare metal in a
container with a 7.68-CPU quota, shared with other tenants (load average about 5.6-5.7 at the
machine check, 5.3-10.6 during the timed rows), CUDA 13.3.73, GCC 13.3.0. It holds summary statistics and
per-row aggregates only, numbers as recorded; it is a screen-grade record (a shared host): the
paired ratios hold, the absolute times are indicative.

## What was measured

All builds are measurement-harness builds of the final core's six sources (`src/core/`) for sm_89
with the 64-block launch plan (`DBENCH_GPU_SM_BUDGET=64`, the plan of the `ada64` core), except
`avx2-b0` and `avx512-b0`, which use the default 142-block launch table. The reference side of every
paired row is `avx512`, the shipped AVX-512 flag set (`-mavx2 -mavx512f -mavx512vl -mavx512dq
-mavx512bw`, with `-ffp-contract=off`). Timed processes run on CPU 0, a primary hardware thread on
the GPU's NUMA node 0, with its SMT sibling (CPU 32) idle, unless a row says otherwise.

| part | design | JSON key |
|---|---|---|
| exactness gate | 7 builds x 15 rows (five problems x `a03125c3`, `a05c11`, `ms30`), 2,700 shots per comparison, per shot against the default AVX-512 build and the ldpc reference: correction, observables, status word and soft output; 0 mismatches | `exactness_gate` |
| host-ISA screens | paired A/A: 3 processes x 2 create orders x {main, A/A}, 20,000 shots per side per run (120,000 pooled per side), both handles in one process, interleaved shot by shot; nearest-rank p50 / p99 / p99.9; verdict by the separation rule against the session's own A/A runs (tuning/KNOWLEDGE.md section 7.1) | `screens` |
| full rows | the same protocol with 4 processes (160,000 shots per side back to back); cadence rows with a 10 ms host busy-wait before every call, without keep-warm and with it (every 500 us and once 25 us before the call), 4,000 shots per side pooled over two sessions | `full_rows` |
| pinning | the `avx512` build A/A in one process under five pinning conditions; per session a discarded warm-up process, then 4 processes x 2 create orders x 20,000 shots per side (320,000 pooled); two rounds in opposite order, each round's ratios against its own idle-sibling row (between-session ratios, not a paired A/B) | `pinning_rows`, `pinning_summary` |
| paired keep-warm | both handles `avx512`, the same shots and gap, only one handle kept warm (2,000 shots per side) | `keep_warm_paired` |
| keep-warm back to back | no gap; keep-warm enqueued after a random half of the pairs, each pair grouped by the treatment of the pair before it | `keep_warm_back_to_back` |
| output invariance | the outputs of the cadence runs compared across runs, builds and keep-warm modes: one distinct result per problem (84 runs on bb144 xyz, 68 on bb288 z) | `keep_warm_output_invariance` |
| two handles in one process | call position and create order of the paired handles (control runs) | `two_handles_call_position_and_create_order` |
| re-measurement | an independent re-measurement of the main rows with the same command and pinned CPU; where it differs from the first measurement, quote it (`corrections_when_quoting`) | `re_measurement` |

The first measurement's AVX2 penalty (screens p50 1.0003-1.0698, p99 1.0137-1.0535) is smaller
than the re-measurement's (bb144 xyz p50 / p99 1.0387 / 1.0764, bb288 z 1.1114 / 1.1344), hence
"about 4-13 % slower" in the documents; the first measurement's 3.4 % keep-warm gain at a 250 us
period did not reproduce (2.6 %).

## Cited by

- `tuning/KNOWLEDGE.md`: section 0.2 (machine M), 2.3 (thread placement, host ISA, keep-warm and
  call-cadence cells), 4.4 (the machine-M paragraph and table), 4.9 (whole call on a Xeon host),
  4.10 (thread placement on a two-socket Xeon host), 6.5 and 6.6 (the machine-M rows), 8 (known
  gaps).
- `docs/SUPPORT.md`, "Host policy".
