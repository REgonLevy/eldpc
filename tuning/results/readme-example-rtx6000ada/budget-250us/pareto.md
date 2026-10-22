# Tuning result

- command: `eldpc-tune --dem model.dem --ler-target 1e-3 --p99-budget-us 250 --auto --out tune250`
- problem: `model.dem`
- shots per point per case: up to 131072 (chunks of 2048)
- confidence of the one-sided bounds: 0.99
- LER target (upper bound per shot): 0.001
- call cadence: 0 us
- latency budget: p99 250 us

## Environment

- `nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv` -> name, driver_version, compute_cap; NVIDIA RTX 6000 Ada Generation, 580.126.09, 8.9
- CUDA toolkit: 13.3 (V13.3.73, from the library's build_info)
- eldpc 0.1.0, host ISA avx2
- CUDA_VISIBLE_DEVICES: (unset); eldpc.devices(): NVIDIA RTX 6000 Ada Generation (cc 8.9, driver 13.0)
- CPU: AMD EPYC 7542 32-Core Processor (128 threads)

## Choice

- config: `bp_method=ms;decoder=bposd;max_iter=31;ms_scaling_factor=0.5;osd_method=osd0;osd_order=0;schedule=parallel`
- LER (accuracy confirmation, worst case): 0.00076; one-sided upper bound 0.000989
- screen (not the reported number): 17/30720 failures, LER 0.000553, upper bound 0.000954
- latency (latency confirmation): p50 183.6 us, p99 217.7 us
- ranking: lowest p50 and p99; candidates within the measured A/A band of the fastest tie
- tie set (1; fastest `bposd:a=0.5:c=11`): within a tie: lower LER upper bound, then fewer BP iterations, then alpha closest to a shipped preset's alpha, then name order
  - `bposd:a=0.5:c=11`: p50 91.80 us, p99 167.69 us, LER upper 0.000954, max_iter 11, alpha 0.5  (screen pick)
- keep-warm: off (call cadence 0 us is at or below 500 us: the clock is held by the call stream itself and the keep-warm tick does not fit before a call)
- confirmation run (8000 calls per case): p50 183.6 us, p99 217.7 us, within budget: yes

## Budget membership

- rule: every screen survivor (a screen pass with --ler-target, a point above the LER floor without) whose latency-stage measurement is within 0.5x-2x of a budget (largest measured/budget ratio) is re-timed before the in/out decision: at least 8000 calls per side and case in a full run, on fresh syndromes, paired with the A/A reference in the ABBA schedule; the decision is that measurement's (with the slow-class rule of a p99 budget); at most 32 points, the nearest to the budget first; outside the margin the latency-stage verdict stands
- reference: `bposd:a=0.5:c=11`; re-timed: 3; decided by the latency stage: 0; capped: 0; wall 5.2 s

| point | latency-stage p99 us (x budget) | re-timed p50 us | re-timed p99 us | reference p99 us | in budget |
|---|---|---|---|---|---|
| `bposd:a=0.5:c=31` | 215.5 (x0.86) | 175.8 | 218.2 | 158.8 | yes |
| `bposd:a=0.5:c=15` | 169.1 (x0.68) | 108.2 | 168.6 | 179.1 | yes |
| `bposd:a=0.5:c=11` | 167.7 (x0.67) | 91.5 | 174.5 | 170.2 | yes |

## Accuracy confirmation

- planned shots: 99992 per candidate and case, declared before the first confirmation shot; no early stop
- shot-count rule: N doubles from 64 until a point whose true LER is target/1.5 (0.000666667) has its one-sided 0.99 upper bound below the target 0.001 with probability >= 0.9, then bisection returns an N with probability >= 0.9 whose N - 1 is below it; the probability is not monotone in N, so a smaller N can also meet the margin
- confidence: 0.99; target: 0.001
- confirmed 3 of 3 in-budget candidate(s), one at a time in the choice's order (stopped at the first accepted candidate; cap: none)
- case seeds: dem -> 7374403776634429647

| candidate | rank | screen upper | confirmation failures/shots | LER | upper | passed | latency within | slow fraction | p99 unreliable | accepted |
|---|---|---|---|---|---|---|---|---|---|---|
| `bposd:a=0.5:c=11` | 1 | 0.000954 | 82/99992 | 0.00082 | 0.00106 | no | yes | 0.769 | no | no |
| `bposd:a=0.5:c=15` | 2 | 0.000954 | 80/99992 | 0.0008 | 0.00103 | no | yes | 0.751 | no | no |
| `bposd:a=0.5:c=31` | 3 | 0.000954 | 76/99992 | 0.00076 | 0.000989 | yes | yes | 0.727 | no | yes |
- `bposd:a=0.5:c=11` (rank 1) not accepted: accuracy: upper bound 0.00106 is not below the target 0.001 (dem: 82/99992 failures)
- `bposd:a=0.5:c=15` (rank 2) not accepted: accuracy: upper bound 0.00103 is not below the target 0.001 (dem: 80/99992 failures)

## Core variants

- device: NVIDIA RTX 6000 Ada Generation (cc 8.9, 142 SMs); the table's variant: ada64
- `bposd:a=0.5:c=31` with default: p50 183.7 us, p99 239.7 us (8000 calls per side and case, paired on the same syndromes)
- `bposd:a=0.5:c=31` with ada64: p50 176.6 us, p99 233.9 us (8000 calls per side and case, paired on the same syndromes)
- ada64 / default: p50 x0.962, p99 x0.976; A/A band p50 +-0.4 %, p99 +-3.0 %
- faster: no difference; keep the table's choice (ada64); no variable needed: the variants do not differ beyond the A/A band
- not written into config.json: Decoder(config=...) has no variant key; the front reads ELDPC_DEVICE_CLASS from the process environment at every create

## Pruning

- rule threshold min_converged 0.99 at confidence 0.99, for a p99 latency budget; enabled: yes
- a point of a (decoder, alpha) row is skipped as dominated by a passing point of the same row with a smaller max_iter only when the one-sided Clopper-Pearson upper bound of the passing point's non-converged share, at the tuner's confidence, is below 1 - min_converged (it is then a tail small enough that the managed latency quantile is set by the converged calls, and raising the cap can only add BP work); while at least 1 - min_converged of the calls take the slow path the quantile sits in the slow class, and a larger cap that moves shots from OSD-0 into the converged class can be faster there
- passing points below the threshold (their larger caps were therefore not pruned): 3 of 3; the lowest converged share is 0.235

## Pareto table

| config | decoder | alpha | max_iter | converged share | LER per case [99% CP interval] (screen) | p50 us | p99 us | in budget | Pareto | note |
|---|---|---|---|---|---|---|---|---|---|---|
| bposd:a=0.5:c=11 | bposd | 0.5 | 11 | 0.235 | dem: 0.000553 [0.00029, 0.000954] | 91.8 | 167.7 | yes | * | pass; initial best; re-timed p50 91.5 / p99 174.5 us (decides 'in budget'); screen pick (not accepted) |
| bposd:a=0.5:c=15 | bposd | 0.5 | 15 | 0.253 | dem: 0.000553 [0.00029, 0.000954] | 108.4 | 169.1 | yes |  | pass; within_noise; re-timed p50 108.2 / p99 168.6 us (decides 'in budget') |
| bposd:a=0.5:c=31 | bposd | 0.5 | 31 | 0.28 | dem: 0.000553 [0.00029, 0.000954] | 175.9 | 215.5 | yes |  | pass; slower; re-timed p50 175.8 / p99 218.2 us (decides 'in budget'); CHOSEN |
| bposd:a=0.25:c=1 | bposd | 0.25 | 1 | 0 | dem: 0.0254 [0.018, 0.0347] | - | - | - |  | fail |
| bposd:a=0.3125:c=1 | bposd | 0.3125 | 1 | 0 | dem: 0.0254 [0.018, 0.0347] | - | - | - |  | fail |
| bposd:a=0.375:c=1 | bposd | 0.375 | 1 | 0 | dem: 0.0249 [0.0176, 0.0341] | - | - | - |  | fail |
| bposd:a=0.5:c=1 | bposd | 0.5 | 1 | 0.00781 | dem: 0.0249 [0.0176, 0.0341] | - | - | - |  | fail |
| bposd:a=0.625:c=1 | bposd | 0.625 | 1 | 0.00684 | dem: 0.0269 [0.0192, 0.0364] | - | - | - |  | fail |
| bposd:a=0.75:c=1 | bposd | 0.75 | 1 | 0.00391 | dem: 0.0273 [0.0196, 0.0369] | - | - | - |  | fail |
| bposd:a=0.25:c=2 | bposd | 0.25 | 2 | 0 | dem: 0.0083 [0.00435, 0.0143] | - | - | - |  | fail |
| bposd:a=0.3125:c=2 | bposd | 0.3125 | 2 | 0 | dem: 0.00537 [0.00233, 0.0105] | - | - | - |  | fail |
| bposd:a=0.375:c=2 | bposd | 0.375 | 2 | 0.000977 | dem: 0.00537 [0.00233, 0.0105] | - | - | - |  | fail |
| bposd:a=0.5:c=2 | bposd | 0.5 | 2 | 0.00879 | dem: 0.00977 [0.00542, 0.0161] | - | - | - |  | fail |
| bposd:a=0.625:c=2 | bposd | 0.625 | 2 | 0.00879 | dem: 0.0303 [0.0221, 0.0403] | - | - | - |  | fail |
| bposd:a=0.75:c=2 | bposd | 0.75 | 2 | 0.00879 | dem: 0.0859 [0.0721, 0.101] | - | - | - |  | fail |
| bposd:a=0.25:c=3 | bposd | 0.25 | 3 | 0 | dem: 0.00488 [0.00202, 0.00981] | - | - | - |  | fail |
| bp:a=0.3125:c=3 | bp | 0.3125 | 3 | 0 | dem: 0.986 [0.979, 0.992] | - | - | - |  | fail |
| bposd:a=0.3125:c=3 | bposd | 0.3125 | 3 | 0 | dem: 0.00342 [0.00114, 0.0078] | - | - | - |  | fail |
| bposd:a=0.375:c=3 | bposd | 0.375 | 3 | 0.00146 | dem: 0.00244 [0.00101, 0.00491] | - | - | - |  | fail |
| bp:a=0.5:c=3 | bp | 0.5 | 3 | 0.019 | dem: 0.933 [0.919, 0.945] | - | - | - |  | fail |
| bposd:a=0.5:c=3 | bposd | 0.5 | 3 | 0.019 | dem: 0.00244 [0.00101, 0.00491] | - | - | - |  | fail |
| bp:a=0.625:c=3 | bp | 0.625 | 3 | 0.0142 | dem: 0.938 [0.924, 0.95] | - | - | - |  | fail |
| bposd:a=0.625:c=3 | bposd | 0.625 | 3 | 0.0142 | dem: 0.00391 [0.00142, 0.00848] | - | - | - |  | fail |
| bposd:a=0.75:c=3 | bposd | 0.75 | 3 | 0.0127 | dem: 0.00537 [0.00233, 0.0105] | - | - | - |  | fail |
| bposd:a=0.25:c=5 | bposd | 0.25 | 5 | 0 | dem: 0.00439 [0.00171, 0.00915] | - | - | - |  | fail |
| bposd:a=0.3125:c=5 | bposd | 0.3125 | 5 | 0 | dem: 0.00342 [0.00114, 0.0078] | - | - | - |  | fail |
| bposd:a=0.375:c=5 | bposd | 0.375 | 5 | 0.00804 | dem: 0.00144 [0.00101, 0.00198] | - | - | - |  | fail |
| bposd:a=0.5:c=5 | bposd | 0.5 | 5 | 0.0291 | dem: 0.00142 [0.00102, 0.00191] | - | - | - |  | fail |
| bposd:a=0.625:c=5 | bposd | 0.625 | 5 | 0.0259 | dem: 0.00208 [0.00109, 0.00358] | - | - | - |  | fail |
| bposd:a=0.75:c=5 | bposd | 0.75 | 5 | 0.0259 | dem: 0.00342 [0.00114, 0.0078] | - | - | - |  | fail |
| bposd:a=0.25:c=7 | bposd | 0.25 | 7 | 0 | dem: 0.00488 [0.00202, 0.00981] | - | - | - |  | fail |
| bp:a=0.3125:c=7 | bp | 0.3125 | 7 | 0 | dem: 0.979 [0.97, 0.986] | - | - | - |  | fail |
| bposd:a=0.3125:c=7 | bposd | 0.3125 | 7 | 0 | dem: 0.00269 [0.00117, 0.00524] | - | - | - |  | fail |
| bposd:a=0.375:c=7 | bposd | 0.375 | 7 | 0.00997 | dem: 0.00138 [0.001, 0.00184] | - | - | - |  | fail |
| bp:a=0.5:c=7 | bp | 0.5 | 7 | 0.0718 | dem: 0.815 [0.794, 0.835] | - | - | - |  | fail |
| bposd:a=0.5:c=7 | bposd | 0.5 | 7 | 0.0642 | dem: 0.00109 [0.00089, 0.00132] | - | - | - |  | inconclusive |
| bp:a=0.625:c=7 | bp | 0.625 | 7 | 0.0347 | dem: 0.896 [0.88, 0.912] | - | - | - |  | fail |
| bposd:a=0.625:c=7 | bposd | 0.625 | 7 | 0.0266 | dem: 0.00135 [0.00101, 0.00177] | - | - | - |  | fail |
| bposd:a=0.75:c=7 | bposd | 0.75 | 7 | 0.0274 | dem: 0.00205 [0.00116, 0.00335] | - | - | - |  | fail |
| bposd:a=0.4375:c=10 | bposd | 0.4375 | 10 | 0.113 | dem: 0.0013 [0.001, 0.00166] | - | - | - |  | fail |
| bposd:a=0.46875:c=10 | bposd | 0.46875 | 10 | 0.195 | dem: 0.00117 [0.000959, 0.00141] | - | - | - |  | inconclusive |
| bposd:a=0.53125:c=10 | bposd | 0.53125 | 10 | 0.0415 | dem: 0.00317 [0.00149, 0.00589] | - | - | - |  | fail |
| bposd:a=0.5625:c=10 | bposd | 0.5625 | 10 | 0.0327 | dem: 0.00488 [0.00202, 0.00981] | - | - | - |  | fail |
| bposd:a=0.25:c=11 | bposd | 0.25 | 11 | 0 | dem: 0.00439 [0.00171, 0.00915] | - | - | - |  | fail |
| bposd:a=0.3125:c=11 | bposd | 0.3125 | 11 | 0 | dem: 0.00244 [0.00101, 0.00491] | - | - | - |  | fail |
| bposd:a=0.375:c=11 | bposd | 0.375 | 11 | 0.0112 | dem: 0.00138 [0.001, 0.00184] | - | - | - |  | fail |
| bposd:a=0.4375:c=11 | bposd | 0.4375 | 11 | 0.113 | dem: 0.00103 [0.000835, 0.00126] | - | - | - |  | inconclusive |
| bposd:a=0.46875:c=11 | bposd | 0.46875 | 11 | 0.205 | dem: 0.000931 [0.000746, 0.00115] | - | - | - |  | inconclusive |
| bposd:a=0.53125:c=11 | bposd | 0.53125 | 11 | 0.0404 | dem: 0.000977 [0.000787, 0.0012] | - | - | - |  | inconclusive |
| bposd:a=0.5625:c=11 | bposd | 0.5625 | 11 | 0.0293 | dem: 0.0011 [0.000897, 0.00133] | - | - | - |  | inconclusive |
| bposd:a=0.625:c=11 | bposd | 0.625 | 11 | 0.0305 | dem: 0.00105 [0.000849, 0.00127] | - | - | - |  | inconclusive |
| bposd:a=0.75:c=11 | bposd | 0.75 | 11 | 0.0289 | dem: 0.00146 [0.00104, 0.002] | - | - | - |  | fail |
| bposd:a=0.25:c=15 | bposd | 0.25 | 15 | 0 | dem: 0.00439 [0.00171, 0.00915] | - | - | - |  | fail |
| bp:a=0.3125:c=15 | bp | 0.3125 | 15 | 0 | dem: 0.979 [0.97, 0.985] | - | - | - |  | fail |
| bposd:a=0.3125:c=15 | bposd | 0.3125 | 15 | 0 | dem: 0.00244 [0.00101, 0.00491] | - | - | - |  | fail |
| bposd:a=0.375:c=15 | bposd | 0.375 | 15 | 0.0121 | dem: 0.00138 [0.001, 0.00184] | - | - | - |  | fail |
| bp:a=0.4375:c=15 | bp | 0.4375 | 15 | 0.106 | dem: 0.831 [0.81, 0.849] | - | - | - |  | fail |
| bp:a=0.46875:c=15 | bp | 0.46875 | 15 | 0.219 | dem: 0.593 [0.568, 0.619] | - | - | - |  | fail |
| bp:a=0.5:c=15 | bp | 0.5 | 15 | 0.255 | dem: 0.544 [0.519, 0.57] | - | - | - |  | fail |
| bp:a=0.53125:c=15 | bp | 0.53125 | 15 | 0.0542 | dem: 0.877 [0.859, 0.893] | - | - | - |  | fail |
| bp:a=0.5625:c=15 | bp | 0.5625 | 15 | 0.0347 | dem: 0.907 [0.891, 0.921] | - | - | - |  | fail |
| bp:a=0.625:c=15 | bp | 0.625 | 15 | 0.04 | dem: 0.898 [0.882, 0.913] | - | - | - |  | fail |
| bposd:a=0.625:c=15 | bposd | 0.625 | 15 | 0.0322 | dem: 0.00108 [0.000883, 0.00131] | - | - | - |  | inconclusive |
| bposd:a=0.75:c=15 | bposd | 0.75 | 15 | 0.0317 | dem: 0.00144 [0.00104, 0.00194] | - | - | - |  | fail |
| bposd:a=0.25:c=31 | bposd | 0.25 | 31 | 0 | dem: 0.00439 [0.00171, 0.00915] | - | - | - |  | fail |
| bp:a=0.3125:c=31 | bp | 0.3125 | 31 | 0 | dem: 0.979 [0.97, 0.985] | - | - | - |  | fail |
| bposd:a=0.3125:c=31 | bposd | 0.3125 | 31 | 0 | dem: 0.00244 [0.00101, 0.00491] | - | - | - |  | fail |
| bposd:a=0.375:c=31 | bposd | 0.375 | 31 | 0.0129 | dem: 0.00138 [0.001, 0.00184] | - | - | - |  | fail |
| bp:a=0.4375:c=31 | bp | 0.4375 | 31 | 0.107 | dem: 0.845 [0.826, 0.863] | - | - | - |  | fail |
| bp:a=0.46875:c=31 | bp | 0.46875 | 31 | 0.237 | dem: 0.574 [0.548, 0.599] | - | - | - |  | fail |
| bp:a=0.5:c=31 | bp | 0.5 | 31 | 0.281 | dem: 0.532 [0.506, 0.558] | - | - | - |  | fail |
| bp:a=0.53125:c=31 | bp | 0.53125 | 31 | 0.0962 | dem: 0.795 [0.774, 0.816] | - | - | - |  | fail |
| bp:a=0.5625:c=31 | bp | 0.5625 | 31 | 0.0386 | dem: 0.901 [0.885, 0.916] | - | - | - |  | fail |
| bp:a=0.625:c=31 | bp | 0.625 | 31 | 0.0425 | dem: 0.896 [0.879, 0.911] | - | - | - |  | fail |
| bposd:a=0.625:c=31 | bposd | 0.625 | 31 | 0.0344 | dem: 0.0013 [0.00101, 0.00165] | - | - | - |  | fail |
| bposd:a=0.75:c=31 | bposd | 0.75 | 31 | 0.037 | dem: 0.00144 [0.00104, 0.00194] | - | - | - |  | fail |

LER is the fraction of shots with at least one wrong observable; intervals are one-sided Clopper-Pearson bounds at the stated confidence.  `*` marks a point on the measured (LER, p50) Pareto frontier.  p50/p99 are single-shot calls through the package API at the stated cadence; the A/A band of the machine is in results.json.
