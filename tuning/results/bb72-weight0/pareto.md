# Tuning result

- command: `eldpc-tune --problem bb72 --p 0.001 --weight 0 --p99-budget-us 1000 --max-shots 2048 --auto --out tuning/results/bb72-weight0`
- problem: `bb72`
- physical error rates: 0.001
- shots per point per case: up to 2048 (chunks of 2048)
- confidence of the one-sided bounds: 0.99
- trade-off weight: 0 (score = log10 LER upper + weight * log10 p50)
- call cadence: 0 us
- latency budget: p99 1000 us

## Environment

- `nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv` -> name, driver_version, compute_cap; NVIDIA GeForce RTX 5080, 595.91.07, 12.0
- `nvcc --version` -> nvcc: NVIDIA (R) Cuda compiler driver; Copyright (c) 2005-2026 NVIDIA Corporation; Built on Tue_Jun_09_02:43:40_PM_PDT_2026; Cuda compilation tools, release 13.3, V13.3.73; Build cuda_13.3.r13.3/compiler.38244171_0
- `$CUDACXX --version` -> nvcc: NVIDIA (R) Cuda compiler driver; Copyright (c) 2005-2026 NVIDIA Corporation; Built on Tue_Jun_09_02:43:40_PM_PDT_2026; Cuda compilation tools, release 13.3, V13.3.73; Build cuda_13.3.r13.3/compiler.38244171_0
- eldpc 0.1.0, host ISA avx512
- CUDA_VISIBLE_DEVICES: (unset); eldpc.devices(): NVIDIA GeForce RTX 5080 (cc 12.0, driver 13.2)
- CPU: AMD Ryzen 9 7950X3D 16-Core Processor (32 threads)

## Choice

- config: `bp_method=ms;decoder=bposd;max_iter=1;ms_scaling_factor=0.375;osd_method=osd0;osd_order=0;schedule=parallel`
- LER (accuracy confirmation, worst case): 0.000977; one-sided upper bound 0.0041
- screen (not the reported number): 0/2048 failures, LER 0, upper bound 0.00225
- latency (latency confirmation): p50 27.6 us, p99 61.3 us
- ranking: score = log10(LER upper bound) + weight * log10(p50_us), smallest wins; scores within weight * log10(1 + A/A p50 band) of the best tie
- tie set (54; fastest `bposd:a=0.5:c=1`): within a tie: lower LER upper bound, then fewer BP iterations, then alpha closest to a shipped preset's alpha, then name order
  - `bposd:a=0.375:c=1`: p50 26.96 us, p99 77.58 us, LER upper 0.00225, max_iter 1, alpha 0.375  <- chosen
  - `bposd:a=0.4375:c=1`: p50 26.73 us, p99 49.73 us, LER upper 0.00225, max_iter 1, alpha 0.4375
  - `bposd:a=0.5:c=1`: p50 26.20 us, p99 58.51 us, LER upper 0.00225, max_iter 1, alpha 0.5
  - `bposd:a=0.625:c=1`: p50 26.55 us, p99 62.26 us, LER upper 0.00225, max_iter 1, alpha 0.625
  - `bposd:a=0.75:c=1`: p50 26.39 us, p99 59.56 us, LER upper 0.00225, max_iter 1, alpha 0.75
  - `bposd:a=0.28125:c=1`: p50 26.82 us, p99 60.65 us, LER upper 0.00225, max_iter 1, alpha 0.28125
  - `bposd:a=0.34375:c=1`: p50 27.01 us, p99 49.14 us, LER upper 0.00225, max_iter 1, alpha 0.34375
  - `bposd:a=0.40625:c=1`: p50 26.84 us, p99 60.72 us, LER upper 0.00225, max_iter 1, alpha 0.40625
  - `bposd:a=0.21875:c=1`: p50 26.78 us, p99 50.43 us, LER upper 0.00225, max_iter 1, alpha 0.21875
  - `bposd:a=0.3125:c=2`: p50 30.53 us, p99 61.13 us, LER upper 0.00225, max_iter 2, alpha 0.3125
  - `bposd:a=0.375:c=2`: p50 30.06 us, p99 60.15 us, LER upper 0.00225, max_iter 2, alpha 0.375
  - `bposd:a=0.4375:c=2`: p50 29.68 us, p99 60.12 us, LER upper 0.00225, max_iter 2, alpha 0.4375
  - `bposd:a=0.5:c=2`: p50 29.89 us, p99 75.86 us, LER upper 0.00225, max_iter 2, alpha 0.5
  - `bposd:a=0.28125:c=2`: p50 30.59 us, p99 61.26 us, LER upper 0.00225, max_iter 2, alpha 0.28125
  - `bposd:a=0.40625:c=2`: p50 29.83 us, p99 60.87 us, LER upper 0.00225, max_iter 2, alpha 0.40625
  - `bposd:a=0.25:c=2`: p50 30.57 us, p99 48.75 us, LER upper 0.00225, max_iter 2, alpha 0.25
  - `bposd:a=0.21875:c=2`: p50 30.59 us, p99 49.09 us, LER upper 0.00225, max_iter 2, alpha 0.21875
  - `bposd:a=0.1875:c=2`: p50 30.53 us, p99 60.88 us, LER upper 0.00225, max_iter 2, alpha 0.1875
  - `bposd:a=0.3125:c=3`: p50 34.49 us, p99 60.53 us, LER upper 0.00225, max_iter 3, alpha 0.3125
  - `bposd:a=0.375:c=3`: p50 34.02 us, p99 53.56 us, LER upper 0.00225, max_iter 3, alpha 0.375
  - `bposd:a=0.5:c=3`: p50 31.21 us, p99 50.94 us, LER upper 0.00225, max_iter 3, alpha 0.5
  - `bposd:a=0.625:c=3`: p50 32.98 us, p99 61.73 us, LER upper 0.00225, max_iter 3, alpha 0.625
  - `bposd:a=0.75:c=3`: p50 31.33 us, p99 61.21 us, LER upper 0.00225, max_iter 3, alpha 0.75
  - `bposd:a=0.25:c=3`: p50 34.59 us, p99 59.74 us, LER upper 0.00225, max_iter 3, alpha 0.25
  - `bposd:a=0.3125:c=5`: p50 42.79 us, p99 63.88 us, LER upper 0.00225, max_iter 5, alpha 0.3125
  - `bposd:a=0.375:c=5`: p50 36.95 us, p99 60.28 us, LER upper 0.00225, max_iter 5, alpha 0.375
  - `bposd:a=0.5:c=5`: p50 31.47 us, p99 53.87 us, LER upper 0.00225, max_iter 5, alpha 0.5
  - `bposd:a=0.625:c=5`: p50 31.46 us, p99 76.41 us, LER upper 0.00225, max_iter 5, alpha 0.625
  - `bposd:a=0.75:c=5`: p50 31.84 us, p99 75.07 us, LER upper 0.00225, max_iter 5, alpha 0.75
  - `bposd:a=0.25:c=5`: p50 42.97 us, p99 61.24 us, LER upper 0.00225, max_iter 5, alpha 0.25
  - `bposd:a=0.3125:c=7`: p50 51.11 us, p99 60.36 us, LER upper 0.00225, max_iter 7, alpha 0.3125
  - `bposd:a=0.375:c=7`: p50 35.89 us, p99 77.15 us, LER upper 0.00225, max_iter 7, alpha 0.375
  - `bposd:a=0.5:c=7`: p50 31.46 us, p99 60.38 us, LER upper 0.00225, max_iter 7, alpha 0.5
  - `bposd:a=0.625:c=7`: p50 48.22 us, p99 60.27 us, LER upper 0.00225, max_iter 7, alpha 0.625
  - `bposd:a=0.75:c=7`: p50 38.36 us, p99 76.03 us, LER upper 0.00225, max_iter 7, alpha 0.75
  - `bposd:a=0.25:c=7`: p50 51.35 us, p99 69.39 us, LER upper 0.00225, max_iter 7, alpha 0.25
  - `bposd:a=0.3125:c=11`: p50 67.90 us, p99 75.64 us, LER upper 0.00225, max_iter 11, alpha 0.3125
  - `bposd:a=0.375:c=11`: p50 39.58 us, p99 76.86 us, LER upper 0.00225, max_iter 11, alpha 0.375
  - `bposd:a=0.5:c=11`: p50 31.69 us, p99 74.11 us, LER upper 0.00225, max_iter 11, alpha 0.5
  - `bposd:a=0.625:c=11`: p50 35.89 us, p99 75.88 us, LER upper 0.00225, max_iter 11, alpha 0.625
  - `bposd:a=0.75:c=11`: p50 31.55 us, p99 73.93 us, LER upper 0.00225, max_iter 11, alpha 0.75
  - `bposd:a=0.25:c=11`: p50 67.98 us, p99 91.97 us, LER upper 0.00225, max_iter 11, alpha 0.25
  - `bposd:a=0.3125:c=15`: p50 84.45 us, p99 92.19 us, LER upper 0.00225, max_iter 15, alpha 0.3125
  - `bposd:a=0.375:c=15`: p50 37.24 us, p99 91.10 us, LER upper 0.00225, max_iter 15, alpha 0.375
  - `bposd:a=0.5:c=15`: p50 31.41 us, p99 87.13 us, LER upper 0.00225, max_iter 15, alpha 0.5
  - `bposd:a=0.625:c=15`: p50 31.91 us, p99 90.08 us, LER upper 0.00225, max_iter 15, alpha 0.625
  - `bposd:a=0.75:c=15`: p50 32.07 us, p99 89.13 us, LER upper 0.00225, max_iter 15, alpha 0.75
  - `bposd:a=0.25:c=15`: p50 84.51 us, p99 94.46 us, LER upper 0.00225, max_iter 15, alpha 0.25
  - `bposd:a=0.3125:c=31`: p50 151.89 us, p99 161.50 us, LER upper 0.00225, max_iter 31, alpha 0.3125
  - `bposd:a=0.375:c=31`: p50 36.55 us, p99 157.38 us, LER upper 0.00225, max_iter 31, alpha 0.375
  - `bposd:a=0.5:c=31`: p50 31.72 us, p99 152.25 us, LER upper 0.00225, max_iter 31, alpha 0.5
  - `bposd:a=0.625:c=31`: p50 44.48 us, p99 156.33 us, LER upper 0.00225, max_iter 31, alpha 0.625
  - `bposd:a=0.75:c=31`: p50 30.51 us, p99 155.66 us, LER upper 0.00225, max_iter 31, alpha 0.75
  - `bposd:a=0.25:c=31`: p50 150.58 us, p99 169.92 us, LER upper 0.00225, max_iter 31, alpha 0.25
- keep-warm: off (call cadence 0 us is at or below 500 us: the clock is held by the call stream itself and the keep-warm tick does not fit before a call)
- confirmation run (8000 calls per case): p50 27.6 us, p99 61.3 us, within budget: yes

## Accuracy confirmation

- planned shots: 2048 per candidate and case, declared before the first confirmation shot; no early stop
- shot-count rule: no --ler-target: a fixed 2048 shots per candidate and case
- confidence: 0.99; target: -; at most 3 candidates, in the choice's order
- case seeds: p=0.001 -> 6283252043746580207

| candidate | rank | screen upper | confirmation failures/shots | LER | upper | passed | latency within | slow fraction | p99 unreliable | accepted |
|---|---|---|---|---|---|---|---|---|---|---|
| `bposd:a=0.375:c=1` | 1 | 0.00225 | 2/2048 | 0.000977 | 0.0041 | yes | yes | 0.876 | no | yes |

## Core variants

- device: NVIDIA GeForce RTX 5080 (cc 12.0, 84 SMs); the table's variant: default
- not measured: this device class has one core variant (default); only compute capability 8.9 with at least 128 SMs has a second (ada64): skipped

## Pruning

- rule threshold min_converged 0.99 at confidence 0.99, for a p99 latency budget; enabled: yes
- a point of a (decoder, alpha) row is skipped as dominated by a passing point of the same row with a smaller max_iter only when the one-sided Clopper-Pearson upper bound of the passing point's non-converged share, at the tuner's confidence, is below 1 - min_converged (it is then a tail small enough that the managed latency quantile is set by the converged calls, and raising the cap can only add BP work); while at least 1 - min_converged of the calls take the slow path the quantile sits in the slow class, and a larger cap that moves shots from OSD-0 into the converged class can be faster there

## Pareto table

| config | decoder | alpha | max_iter | converged share | LER per case [99% CP interval] (screen) | p50 us | p99 us | in budget | Pareto | note |
|---|---|---|---|---|---|---|---|---|---|---|
| bposd:a=0.5:c=1 | bposd | 0.5 | 1 | 0.473 | p=0.001: 0 [0, 0.00225] | 26.2 | 58.5 | yes | * | measured; within_noise |
| bposd:a=0.75:c=1 | bposd | 0.75 | 1 | 0.417 | p=0.001: 0 [0, 0.00225] | 26.4 | 59.6 | yes |  | measured; slower |
| bposd:a=0.625:c=1 | bposd | 0.625 | 1 | 0.473 | p=0.001: 0 [0, 0.00225] | 26.6 | 62.3 | yes |  | measured; initial best |
| bposd:a=0.1875:c=1 | bposd | 0.1875 | 1 | 0.0854 | p=0.001: 0.000488 [4.91e-06, 0.00324] | 26.7 | 60.2 | yes |  | measured; within_noise |
| bposd:a=0.4375:c=1 | bposd | 0.4375 | 1 | 0.188 | p=0.001: 0 [0, 0.00225] | 26.7 | 49.7 | yes |  | measured; within_noise |
| bposd:a=0.25:c=1 | bposd | 0.25 | 1 | 0.084 | p=0.001: 0.000977 [7.26e-05, 0.0041] | 26.8 | 60.9 | yes |  | measured; slower |
| bposd:a=0.21875:c=1 | bposd | 0.21875 | 1 | 0.0854 | p=0.001: 0 [0, 0.00225] | 26.8 | 50.4 | yes |  | measured; within_noise |
| bposd:a=0.28125:c=1 | bposd | 0.28125 | 1 | 0.109 | p=0.001: 0 [0, 0.00225] | 26.8 | 60.7 | yes |  | measured; within_noise |
| bposd:a=0.40625:c=1 | bposd | 0.40625 | 1 | 0.19 | p=0.001: 0 [0, 0.00225] | 26.8 | 60.7 | yes |  | measured; within_noise |
| bposd:a=0.3125:c=1 | bposd | 0.3125 | 1 | 0.0967 | p=0.001: 0.000488 [4.91e-06, 0.00324] | 26.9 | 46.7 | yes |  | measured; within_noise |
| bposd:a=0.375:c=1 | bposd | 0.375 | 1 | 0.121 | p=0.001: 0 [0, 0.00225] | 27.0 | 77.6 | yes |  | measured; slower; CHOSEN |
| bposd:a=0.34375:c=1 | bposd | 0.34375 | 1 | 0.118 | p=0.001: 0 [0, 0.00225] | 27.0 | 49.1 | yes |  | measured; within_noise |
| bp:a=0.625:c=3 | bp | 0.625 | 3 | 0.51 | p=0.001: 0.354 [0.329, 0.379] | 27.0 | 39.7 | yes |  | measured; within_noise |
| bp:a=0.5:c=3 | bp | 0.5 | 3 | 0.565 | p=0.001: 0.311 [0.287, 0.335] | 27.6 | 44.9 | yes |  | measured; within_noise |
| bp:a=0.53125:c=31 | bp | 0.53125 | 31 | 0.605 | p=0.001: 0.262 [0.239, 0.285] | 29.0 | 152.1 | yes |  | measured; slower |
| bp:a=0.4375:c=31 | bp | 0.4375 | 31 | 0.853 | p=0.001: 0.11 [0.0943, 0.127] | 29.1 | 150.1 | yes |  | measured; slower |
| bp:a=0.3125:c=3 | bp | 0.3125 | 3 | 0.201 | p=0.001: 0.489 [0.463, 0.515] | 29.1 | 60.6 | yes |  | measured; within_noise |
| bp:a=0.5:c=31 | bp | 0.5 | 31 | 0.879 | p=0.001: 0.0688 [0.0564, 0.083] | 29.1 | 151.3 | yes |  | measured; slower |
| bp:a=0.5:c=15 | bp | 0.5 | 15 | 0.874 | p=0.001: 0.0737 [0.0609, 0.0883] | 29.1 | 86.6 | yes |  | measured; slower |
| bp:a=0.5:c=7 | bp | 0.5 | 7 | 0.587 | p=0.001: 0.294 [0.271, 0.318] | 29.1 | 60.1 | yes |  | measured; within_noise |
| bp:a=0.4375:c=15 | bp | 0.4375 | 15 | 0.852 | p=0.001: 0.111 [0.0957, 0.129] | 29.1 | 87.5 | yes |  | measured; slower |
| bp:a=0.46875:c=31 | bp | 0.46875 | 31 | 0.863 | p=0.001: 0.0869 [0.073, 0.102] | 29.1 | 149.9 | yes |  | measured; slower |
| bposd:a=0.75:c=2 | bposd | 0.75 | 2 | 0.498 | p=0.001: 0.00928 [0.00506, 0.0155] | 29.4 | 51.5 | yes |  | measured; within_noise |
| bposd:a=0.4375:c=2 | bposd | 0.4375 | 2 | 0.495 | p=0.001: 0 [0, 0.00225] | 29.7 | 60.1 | yes |  | measured; within_noise |
| bposd:a=0.625:c=2 | bposd | 0.625 | 2 | 0.495 | p=0.001: 0.00537 [0.00233, 0.0105] | 29.8 | 77.7 | yes |  | measured; within_noise |
| bposd:a=0.40625:c=2 | bposd | 0.40625 | 2 | 0.48 | p=0.001: 0 [0, 0.00225] | 29.8 | 60.9 | yes |  | measured; within_noise |
| bposd:a=0.5:c=2 | bposd | 0.5 | 2 | 0.49 | p=0.001: 0 [0, 0.00225] | 29.9 | 75.9 | yes |  | measured; slower |
| bp:a=0.46875:c=15 | bp | 0.46875 | 15 | 0.861 | p=0.001: 0.0889 [0.0748, 0.105] | 30.0 | 86.6 | yes |  | measured; within_noise |
| bposd:a=0.375:c=2 | bposd | 0.375 | 2 | 0.38 | p=0.001: 0 [0, 0.00225] | 30.1 | 60.2 | yes |  | measured; within_noise |
| bposd:a=0.34375:c=2 | bposd | 0.34375 | 2 | 0.213 | p=0.001: 0.000488 [4.91e-06, 0.00324] | 30.4 | 61.2 | yes |  | measured; within_noise |
| bposd:a=0.75:c=31 | bposd | 0.75 | 31 | 0.551 | p=0.001: 0 [0, 0.00225] | 30.5 | 155.7 | yes |  | measured; slower |
| bposd:a=0.1875:c=2 | bposd | 0.1875 | 2 | 0.0854 | p=0.001: 0 [0, 0.00225] | 30.5 | 60.9 | yes |  | measured; within_noise |
| bposd:a=0.3125:c=2 | bposd | 0.3125 | 2 | 0.152 | p=0.001: 0 [0, 0.00225] | 30.5 | 61.1 | yes |  | measured; within_noise |
| bposd:a=0.25:c=2 | bposd | 0.25 | 2 | 0.0957 | p=0.001: 0 [0, 0.00225] | 30.6 | 48.8 | yes |  | measured; within_noise |
| bposd:a=0.28125:c=2 | bposd | 0.28125 | 2 | 0.117 | p=0.001: 0 [0, 0.00225] | 30.6 | 61.3 | yes |  | measured; within_noise |
| bposd:a=0.21875:c=2 | bposd | 0.21875 | 2 | 0.0854 | p=0.001: 0 [0, 0.00225] | 30.6 | 49.1 | yes |  | measured; within_noise |
| bposd:a=0.5:c=3 | bposd | 0.5 | 3 | 0.565 | p=0.001: 0 [0, 0.00225] | 31.2 | 50.9 | yes |  | measured; within_noise |
| bposd:a=0.75:c=3 | bposd | 0.75 | 3 | 0.521 | p=0.001: 0 [0, 0.00225] | 31.3 | 61.2 | yes |  | measured; within_noise |
| bposd:a=0.5:c=15 | bposd | 0.5 | 15 | 0.874 | p=0.001: 0 [0, 0.00225] | 31.4 | 87.1 | yes |  | measured; slower |
| bposd:a=0.625:c=5 | bposd | 0.625 | 5 | 0.521 | p=0.001: 0 [0, 0.00225] | 31.5 | 76.4 | yes |  | measured; slower |
| bposd:a=0.5:c=7 | bposd | 0.5 | 7 | 0.587 | p=0.001: 0 [0, 0.00225] | 31.5 | 60.4 | yes |  | measured; within_noise |
| bposd:a=0.5:c=5 | bposd | 0.5 | 5 | 0.582 | p=0.001: 0 [0, 0.00225] | 31.5 | 53.9 | yes |  | measured; within_noise |
| bposd:a=0.75:c=11 | bposd | 0.75 | 11 | 0.547 | p=0.001: 0 [0, 0.00225] | 31.6 | 73.9 | yes |  | measured; within_noise |
| bposd:a=0.5:c=11 | bposd | 0.5 | 11 | 0.869 | p=0.001: 0 [0, 0.00225] | 31.7 | 74.1 | yes |  | measured; within_noise |
| bposd:a=0.5:c=31 | bposd | 0.5 | 31 | 0.879 | p=0.001: 0 [0, 0.00225] | 31.7 | 152.2 | yes |  | measured; slower |
| bposd:a=0.75:c=5 | bposd | 0.75 | 5 | 0.536 | p=0.001: 0 [0, 0.00225] | 31.8 | 75.1 | yes |  | measured; within_noise |
| bposd:a=0.625:c=15 | bposd | 0.625 | 15 | 0.55 | p=0.001: 0 [0, 0.00225] | 31.9 | 90.1 | yes |  | measured; slower |
| bposd:a=0.75:c=15 | bposd | 0.75 | 15 | 0.549 | p=0.001: 0 [0, 0.00225] | 32.1 | 89.1 | yes |  | measured; slower |
| bposd:a=0.625:c=3 | bposd | 0.625 | 3 | 0.51 | p=0.001: 0 [0, 0.00225] | 33.0 | 61.7 | yes |  | measured; slower |
| bposd:a=0.375:c=3 | bposd | 0.375 | 3 | 0.437 | p=0.001: 0 [0, 0.00225] | 34.0 | 53.6 | yes |  | measured; within_noise |
| bposd:a=0.3125:c=3 | bposd | 0.3125 | 3 | 0.201 | p=0.001: 0 [0, 0.00225] | 34.5 | 60.5 | yes |  | measured; within_noise |
| bposd:a=0.25:c=3 | bposd | 0.25 | 3 | 0.0967 | p=0.001: 0 [0, 0.00225] | 34.6 | 59.7 | yes |  | measured; within_noise |
| bposd:a=0.375:c=7 | bposd | 0.375 | 7 | 0.562 | p=0.001: 0 [0, 0.00225] | 35.9 | 77.2 | yes |  | measured; slower |
| bposd:a=0.625:c=11 | bposd | 0.625 | 11 | 0.549 | p=0.001: 0 [0, 0.00225] | 35.9 | 75.9 | yes |  | measured; within_noise |
| bposd:a=0.375:c=31 | bposd | 0.375 | 31 | 0.569 | p=0.001: 0 [0, 0.00225] | 36.5 | 157.4 | yes |  | measured; slower |
| bposd:a=0.375:c=5 | bposd | 0.375 | 5 | 0.553 | p=0.001: 0 [0, 0.00225] | 36.9 | 60.3 | yes |  | measured; within_noise |
| bp:a=0.5625:c=15 | bp | 0.5625 | 15 | 0.551 | p=0.001: 0.305 [0.282, 0.329] | 37.1 | 87.8 | yes |  | measured; slower |
| bposd:a=0.375:c=15 | bposd | 0.375 | 15 | 0.568 | p=0.001: 0 [0, 0.00225] | 37.2 | 91.1 | yes |  | measured; slower |
| bp:a=0.5625:c=31 | bp | 0.5625 | 31 | 0.554 | p=0.001: 0.304 [0.28, 0.328] | 37.3 | 152.1 | yes |  | measured; slower |
| bp:a=0.625:c=7 | bp | 0.625 | 7 | 0.525 | p=0.001: 0.309 [0.285, 0.333] | 37.3 | 77.2 | yes |  | measured; slower |
| bp:a=0.625:c=31 | bp | 0.625 | 31 | 0.553 | p=0.001: 0.325 [0.301, 0.349] | 37.4 | 153.1 | yes |  | measured; slower |
| bp:a=0.625:c=15 | bp | 0.625 | 15 | 0.55 | p=0.001: 0.326 [0.302, 0.351] | 37.4 | 88.5 | yes |  | measured; slower |
| bp:a=0.53125:c=15 | bp | 0.53125 | 15 | 0.598 | p=0.001: 0.267 [0.244, 0.29] | 37.4 | 87.6 | yes |  | measured; within_noise |
| bposd:a=0.75:c=7 | bposd | 0.75 | 7 | 0.542 | p=0.001: 0 [0, 0.00225] | 38.4 | 76.0 | yes |  | measured; slower |
| bposd:a=0.375:c=11 | bposd | 0.375 | 11 | 0.565 | p=0.001: 0 [0, 0.00225] | 39.6 | 76.9 | yes |  | measured; slower |
| bposd:a=0.3125:c=5 | bposd | 0.3125 | 5 | 0.208 | p=0.001: 0 [0, 0.00225] | 42.8 | 63.9 | yes |  | measured; within_noise |
| bposd:a=0.25:c=5 | bposd | 0.25 | 5 | 0.0967 | p=0.001: 0 [0, 0.00225] | 43.0 | 61.2 | yes |  | measured; within_noise |
| bposd:a=0.625:c=31 | bposd | 0.625 | 31 | 0.553 | p=0.001: 0 [0, 0.00225] | 44.5 | 156.3 | yes |  | measured; slower |
| bposd:a=0.625:c=7 | bposd | 0.625 | 7 | 0.525 | p=0.001: 0 [0, 0.00225] | 48.2 | 60.3 | yes |  | measured; within_noise |
| bp:a=0.3125:c=7 | bp | 0.3125 | 7 | 0.21 | p=0.001: 0.465 [0.44, 0.491] | 49.5 | 61.6 | yes |  | measured; within_noise |
| bposd:a=0.3125:c=7 | bposd | 0.3125 | 7 | 0.21 | p=0.001: 0 [0, 0.00225] | 51.1 | 60.4 | yes |  | measured; within_noise |
| bposd:a=0.25:c=7 | bposd | 0.25 | 7 | 0.0967 | p=0.001: 0 [0, 0.00225] | 51.3 | 69.4 | yes |  | measured; slower |
| bposd:a=0.3125:c=11 | bposd | 0.3125 | 11 | 0.211 | p=0.001: 0 [0, 0.00225] | 67.9 | 75.6 | yes |  | measured; slower |
| bposd:a=0.25:c=11 | bposd | 0.25 | 11 | 0.0967 | p=0.001: 0 [0, 0.00225] | 68.0 | 92.0 | yes |  | measured; slower |
| bposd:a=0.3125:c=15 | bposd | 0.3125 | 15 | 0.211 | p=0.001: 0 [0, 0.00225] | 84.4 | 92.2 | yes |  | measured; within_noise |
| bposd:a=0.25:c=15 | bposd | 0.25 | 15 | 0.0967 | p=0.001: 0 [0, 0.00225] | 84.5 | 94.5 | yes |  | measured; slower |
| bp:a=0.3125:c=15 | bp | 0.3125 | 15 | 0.211 | p=0.001: 0.464 [0.438, 0.49] | 86.4 | 100.2 | yes |  | measured; slower |
| bp:a=0.3125:c=31 | bp | 0.3125 | 31 | 0.211 | p=0.001: 0.464 [0.438, 0.49] | 147.8 | 152.2 | yes |  | measured; slower |
| bposd:a=0.25:c=31 | bposd | 0.25 | 31 | 0.0967 | p=0.001: 0 [0, 0.00225] | 150.6 | 169.9 | yes |  | measured; slower |
| bposd:a=0.3125:c=31 | bposd | 0.3125 | 31 | 0.211 | p=0.001: 0 [0, 0.00225] | 151.9 | 161.5 | yes |  | measured; slower |

LER is the fraction of shots with at least one wrong observable; intervals are one-sided Clopper-Pearson bounds at the stated confidence.  `*` marks a point on the measured (LER, p50) Pareto frontier.  p50/p99 are single-shot calls through the package API at the stated cadence; the A/A band of the machine is in results.json.
