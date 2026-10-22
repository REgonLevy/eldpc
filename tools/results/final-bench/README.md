# Final bench of the shipped core on machine A

`final-bench.json`: the pre-registered latency bench of the shipped decoder core (build 6g, the frozen
`src/core/`) on machine A of [tuning/KNOWLEDGE.md](../../../tuning/KNOWLEDGE.md) section 0.2: an RTX 5080
(84 SMs, 64 MiB L2, driver 615.71.09) with a Ryzen 7 9850X3D host in bench mode (isolated CPUs, each timed
process alone on one of them, performance governor), at two GPU profiles: `oc` (+400 MHz graphics, +3,000 MHz
memory, 396 W power limit) and `stock` (360 W). Summary statistics only, as recorded (µs, nearest-rank
quantiles).

How it was produced: the measurement harness timed one C decode call per shot on fresh shot sets of
bb144 xyz, bb144 z and bb288 z (`*_unif_p1e-3_*`):

- part P: the cap curve at alpha 0.5 (caps 3-21), one handle per process, back to back, 3 processes x 20,000 shots;
- part S: the presets (and builds 6e and 6f at `a03125c3`), 4 processes x 20,000 shots;
- parts A and B: paired runs, candidate and reference in one process on the same shots, interleaved, the create
  order alternating; 6g against 6e, against 6f, and against a second copy of itself (the A/A control that gives
  the A/A band); 20,000 shots per side per process;
- parts D and G: bb144 xyz `a03125c3`, 6 repetitions of 2,000 calls with a 0, 10 µs or 10 ms busy-wait gap
  before each call, with `keep_warm()` every 0.5 ms through the 10 ms gap or without it.

Per profile: 221 processes, 0 failed; 2.6M paired shots with 0 correction or observable mismatches; output
checksums equal across the cadence conditions; 0 Xid lines in the session.

Cited by: tuning/KNOWLEDGE.md sections 0.2, 1.3, 1.5, 2.3, 3.1, 3.2, 4.1, 4.4, 4.5, 5.2-5.5 and 7.1;
tuning/TUNER.md section 3; README.md ("Results"); docs/CLAIMS.md.
