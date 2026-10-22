# Tuning result

- command: `eldpc-tune --dem docs/results/tuning-bb144/model.dem --ler-target 0.005 --p99-budget-us 500 --auto --quick --out docs/results/tuning-bb144`
- problem: `docs/results/tuning-bb144/model.dem`
- shots per point per case: up to 2048 (chunks of 512)
- confidence of the one-sided bounds: 0.99
- LER target (upper bound per shot): 0.005
- call cadence: 0 us
- latency budget: p99 500 us

## Environment

- `nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv` -> name, driver_version, compute_cap; NVIDIA RTX 6000 Ada Generation, 580.126.09, 8.9
- CUDA toolkit: 13.3 (V13.3.73, from the library's build_info)
- eldpc 0.1.0, host ISA avx2
- CUDA_VISIBLE_DEVICES: (unset); eldpc.devices(): NVIDIA RTX 6000 Ada Generation (cc 8.9, driver 13.0)
- CPU: AMD EPYC 7542 32-Core Processor (128 threads)

## Choice

- config: `bp_method=ms;decoder=bposd;max_iter=1;ms_scaling_factor=0.53125;osd_method=osd0;osd_order=0;schedule=parallel`
- LER (accuracy confirmation, worst case): 0.000488; one-sided upper bound 0.00324
- screen (not the reported number): 2/2048 failures, LER 0.000977, upper bound 0.0041
- latency (latency confirmation): p50 64.9 us, p99 147.8 us
- ranking: lowest p50 and p99; candidates within the measured A/A band of the fastest tie
- tie set (3; fastest `bposd:a=0.5625:c=1`): within a tie: lower LER upper bound, then fewer BP iterations, then alpha closest to a shipped preset's alpha, then name order
  - `bposd:a=0.53125:c=1`: p50 64.23 us, p99 150.74 us, LER upper 0.0041, max_iter 1, alpha 0.53125  <- chosen
  - `bposd:a=0.5625:c=1`: p50 63.94 us, p99 146.09 us, LER upper 0.0041, max_iter 1, alpha 0.5625
  - `bposd:a=0.4375:c=1`: p50 63.95 us, p99 143.70 us, LER upper 0.0049, max_iter 1, alpha 0.4375
- keep-warm: off (call cadence 0 us is at or below 500 us: the clock is held by the call stream itself and the keep-warm tick does not fit before a call)
- confirmation run (400 calls per case): p50 64.9 us, p99 147.8 us, within budget: yes

## Budget membership

- rule: every screen survivor (a screen pass with --ler-target, a point above the LER floor without) whose latency-stage measurement is within 0.5x-2x of a budget (largest measured/budget ratio) is re-timed before the in/out decision: at least 8000 calls per side and case in a full run, on fresh syndromes, paired with the A/A reference in the ABBA schedule; the decision is that measurement's (with the slow-class rule of a p99 budget); at most 32 points, the nearest to the budget first; outside the margin the latency-stage verdict stands
- reference: `bposd:a=0.5625:c=1`; re-timed: 2; decided by the latency stage: 13; capped: 0; wall 8.6 s

| point | latency-stage p99 us (x budget) | re-timed p50 us | re-timed p99 us | reference p99 us | in budget |
|---|---|---|---|---|---|
| `bposd:a=0.625:c=15` | 300.4 (x0.60) | 217.0 | 273.3 | 169.0 | yes |
| `bposd:a=0.5:c=15` | 258.5 (x0.52) | 216.2 | 302.0 | 152.3 | yes |

## Accuracy confirmation

- planned shots: 2048 per candidate and case, declared before the first confirmation shot; no early stop
- shot-count rule: N doubles from 64 until a point whose true LER is target/1.5 (0.00333333) has its one-sided 0.99 upper bound below the target 0.005 with probability >= 0.9, then bisection returns an N with probability >= 0.9 whose N - 1 is below it; the probability is not monotone in N, so a smaller N can also meet the margin; --quick caps N at 2048
- confidence: 0.99; target: 0.005
- confirmed 1 of 15 in-budget candidate(s), one at a time in the choice's order (stopped at the first accepted candidate; cap: none)
- case seeds: dem -> 701169861627622127

| candidate | rank | screen upper | confirmation failures/shots | LER | upper | passed | latency within | slow fraction | p99 unreliable | accepted |
|---|---|---|---|---|---|---|---|---|---|---|
| `bposd:a=0.53125:c=1` | 1 | 0.0041 | 1/2048 | 0.000488 | 0.00324 | yes | yes | 1 | no | yes |

## Core variants

- device: NVIDIA RTX 6000 Ada Generation (cc 8.9, 142 SMs); the table's variant: ada64
- `bposd:a=0.53125:c=1` with default: p50 97.8 us, p99 185.4 us (1600 calls per side and case, paired on the same syndromes)
- `bposd:a=0.53125:c=1` with ada64: p50 64.7 us, p99 153.8 us (1600 calls per side and case, paired on the same syndromes)
- ada64 / default: p50 x0.661, p99 x0.829; A/A band p50 +-1.1 %, p99 +-2.8 %
- faster: ada64; keep the table's choice (ada64); no variable needed
- not written into config.json: Decoder(config=...) has no variant key; the front reads ELDPC_DEVICE_CLASS from the process environment at every create

## Pruning

- rule threshold min_converged 0.99 at confidence 0.99, for a p99 latency budget; enabled: yes
- a point of a (decoder, alpha) row is skipped as dominated by a passing point of the same row with a smaller max_iter only when the one-sided Clopper-Pearson upper bound of the passing point's non-converged share, at the tuner's confidence, is below 1 - min_converged (it is then a tail small enough that the managed latency quantile is set by the converged calls, and raising the cap can only add BP work); while at least 1 - min_converged of the calls take the slow path the quantile sits in the slow class, and a larger cap that moves shots from OSD-0 into the converged class can be faster there
- passing points below the threshold (their larger caps were therefore not pruned): 15 of 15; the lowest converged share is 0

## Pareto table

| config | decoder | alpha | max_iter | converged share | LER per case [99% CP interval] (screen) | p50 us | p99 us | in budget | Pareto | note |
|---|---|---|---|---|---|---|---|---|---|---|
| bposd:a=0.5625:c=1 | bposd | 0.5625 | 1 | 0.000488 | dem: 0.000977 [7.26e-05, 0.0041] | 63.9 | 146.1 | yes | * | pass; initial best |
| bposd:a=0.4375:c=1 | bposd | 0.4375 | 1 | 0 | dem: 0.00146 [0.000213, 0.0049] | 63.9 | 143.7 | yes |  | pass; within_noise |
| bposd:a=0.46875:c=1 | bposd | 0.46875 | 1 | 0 | dem: 0.00146 [0.000213, 0.0049] | 64.0 | 152.9 | yes |  | pass; within_noise |
| bposd:a=0.53125:c=1 | bposd | 0.53125 | 1 | 0.000488 | dem: 0.000977 [7.26e-05, 0.0041] | 64.2 | 150.7 | yes |  | pass; within_noise; CHOSEN |
| bposd:a=0.5:c=1 | bposd | 0.5 | 1 | 0 | dem: 0 [0, 0.00449] | 64.4 | 183.5 | yes |  | pass; within_noise |
| bposd:a=0.625:c=1 | bposd | 0.625 | 1 | 0 | dem: 0 [0, 0.00449] | 65.2 | 162.4 | yes |  | pass; within_noise |
| bposd:a=0.3125:c=3 | bposd | 0.3125 | 3 | 0 | dem: 0 [0, 0.00449] | 83.4 | 165.9 | yes |  | pass; within_noise |
| bposd:a=0.5:c=3 | bposd | 0.5 | 3 | 0.000977 | dem: 0 [0, 0.00449] | 85.0 | 142.9 | yes |  | pass; within_noise |
| bposd:a=0.625:c=3 | bposd | 0.625 | 3 | 0.000977 | dem: 0 [0, 0.00449] | 88.5 | 174.7 | yes |  | pass; slower |
| bposd:a=0.3125:c=7 | bposd | 0.3125 | 7 | 0.000977 | dem: 0 [0, 0.00449] | 127.5 | 183.6 | yes |  | pass; slower |
| bposd:a=0.5:c=7 | bposd | 0.5 | 7 | 0.0117 | dem: 0 [0, 0.00449] | 128.2 | 164.8 | yes |  | pass; within_noise |
| bposd:a=0.625:c=7 | bposd | 0.625 | 7 | 0.00488 | dem: 0 [0, 0.00449] | 129.3 | 222.4 | yes |  | pass; slower |
| bposd:a=0.3125:c=15 | bposd | 0.3125 | 15 | 0.000977 | dem: 0 [0, 0.00449] | 215.5 | 246.2 | yes |  | pass; slower |
| bposd:a=0.5:c=15 | bposd | 0.5 | 15 | 0.0371 | dem: 0 [0, 0.00449] | 216.0 | 258.5 | yes |  | pass; slower; re-timed p50 216.2 / p99 302.0 us (decides 'in budget') |
| bposd:a=0.625:c=15 | bposd | 0.625 | 15 | 0.0195 | dem: 0 [0, 0.00449] | 217.2 | 300.4 | yes |  | pass; slower; re-timed p50 217.0 / p99 273.3 us (decides 'in budget') |
| bposd:a=0.3125:c=1 | bposd | 0.3125 | 1 | 0 | dem: 0.00244 [0.000625, 0.00639] | - | - | - |  | inconclusive |
| bp:a=0.25:c=3 | bp | 0.25 | 3 | 0 | dem: 0.971 [0.948, 0.985] | - | - | - |  | fail |
| bp:a=0.28125:c=3 | bp | 0.28125 | 3 | 0 | dem: 0.986 [0.969, 0.995] | - | - | - |  | fail |
| bp:a=0.3125:c=3 | bp | 0.3125 | 3 | 0 | dem: 0.957 [0.931, 0.975] | - | - | - |  | fail |
| bp:a=0.5:c=3 | bp | 0.5 | 3 | 0 | dem: 0.984 [0.966, 0.994] | - | - | - |  | fail |
| bp:a=0.25:c=15 | bp | 0.25 | 15 | 0 | dem: 0.967 [0.943, 0.983] | - | - | - |  | fail |
| bp:a=0.28125:c=15 | bp | 0.28125 | 15 | 0 | dem: 0.955 [0.929, 0.974] | - | - | - |  | fail |
| bp:a=0.3125:c=15 | bp | 0.3125 | 15 | 0.00195 | dem: 0.932 [0.901, 0.955] | - | - | - |  | fail |
| bp:a=0.5:c=15 | bp | 0.5 | 15 | 0.0312 | dem: 0.91 [0.876, 0.937] | - | - | - |  | fail |

LER is the fraction of shots with at least one wrong observable; intervals are one-sided Clopper-Pearson bounds at the stated confidence.  `*` marks a point on the measured (LER, p50) Pareto frontier.  p50/p99 are single-shot calls through the package API at the stated cadence; the A/A band of the machine is in results.json.
