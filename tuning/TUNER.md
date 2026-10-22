# Tuning eldpc for a code, a noise level or a machine

Tuning chooses the decoder configuration (min-sum scaling `ms_scaling_factor` = alpha, iteration cap
`max_iter`, BP+OSD-0 or BP alone) and the platform settings (cadence, keep-warm, pinning, clocks) for
one code, its noise and one machine. Outputs are exact, so an LER measured on any machine holds on
every machine for the same config and shots; only latency must be measured on the deployment machine
([KNOWLEDGE.md](KNOWLEDGE.md) 0.1). Run every command from the repository root. Read KNOWLEDGE.md
section 0 before you start.

## 1. Collect the inputs

Write these down first; the report (section 9) repeats them.

- **The problem**: a stim circuit, a stim detector error model (DEM), or a parity-check matrix H with
  its observable matrix O (code capacity). Prefer **one DEM file per noise point**, used by every step
  (tuner, baseline, benchmark). The tuner's built-in recipes and `--circuit` derive the DEM
  **undecomposed** by default: `decompose_errors=False, approximate_disjoint_errors=True`. That is the
  model BP+OSD decodes directly -- one column per error instruction of the flattened DEM; decomposing
  the errors gives a different mechanism list, i.e. a different model, so `--decompose` (which adds
  `ignore_decomposition_failures=True`) changes the problem and is not the default; `--no-decompose`
  restates the default. To write a DEM file from a circuit the same way:
  `str(circuit.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True))`.
  `--dem FILE` takes a DEM as it is and reports `dem_options: null` for it. For a CSS half (for
  example Hx) you also need O: observables derived from H alone are right only for the full
  commuting stabilizer matrix (tuning/README.md).
  Note m, n, nnz(H), max row degree, k, rounds and basis (z-type or joint X+Z; `eldpc.dem_to_arrays`
  returns H and O): they set the create-time shape and the cost per iteration (KNOWLEDGE.md 2.5, 3.3).
- **The noise points**: every physical error rate the deployment must cover; two or more when it varies.
- **The latency budget**: quantile (p50 or p99), value in µs, API (single-shot C call or Python
  `Decoder.decode`), and call cadence (back to back, or the gap between calls).
  p99 budgets are host-dependent: measured on an RTX 6000 Ada with a two-socket EPYC 7542 host (Zen 2),
  1.0-1.2 % of Python `decode_full` calls carried an extra 65-99 µs host tail (median about 80 µs) at
  every configuration, so the p99 sat at about p50 + 75 µs whatever the decoder did (KNOWLEDGE.md 7.1,
  development record): set and tune a p99 budget on the deployment machine.
- **The accuracy requirement**: an LER bar per noise point (section 4), or a trade-off weight.
- **The machine**: GPU model, SM count, L2, driver, CUDA toolkit, clock profile (stock, or the exact
  overclock offsets and power limit), CPU model, and whether anything else uses it.

## 2. Install and check exactness on this machine

```bash
python -m venv .venv && . .venv/bin/activate
pip install ".[test,qldpc]"                     # docs/BUILD.md: toolchain, extras, CUDA runtime, options
python -c "import eldpc; print(eldpc.build_info()); print(eldpc.isa()); print(eldpc.devices())"
(cd src/core && sha256sum -c SOURCES.sha256)    # the frozen core sources are unmodified (text without comments: python tools/check_comment_only.py --check)
ELDPC_REQUIRE_GPU=1 python -m pytest tests      # a missing GPU or CUDA runtime fails, not skips
python -m pytest tuning/tests
python -m pytest -q bench/tests
python verify/run_verify.py --quick --out verify-quick
```

On an offline machine, or in a prepared environment where the build dependencies (CMake, ninja,
scikit-build-core, a C++17 compiler, `nvcc`) and the runtime packages (numpy, scipy, stim) are
already installed, install the source tree without contacting an index and without build
isolation:

```bash
python -m pip install --no-build-isolation --no-deps .
```

- `run_verify.py` must print `PASS` and exit 0 (bit identity to the reference BP+OSD-0 on this
  machine, docs/VERIFY.md); otherwise stop. It needs qLDPC (as do the tuner's `--problem` recipes;
  the `qldpc` extra installs the tested revision), CMake >= 3.16 and a C++17 compiler.
- `eldpc.devices()` must show compute capability 8.9 or 12.0, the only architectures the build carries;
  note `sm_count` and `l2_bytes`. The decoder uses device 0 of the visible devices: choose the GPU with
  `CUDA_VISIBLE_DEVICES=N`. `eldpc.isa()` names the host core loaded; both cores are bit-identical.
- Another GPU architecture needs the list in `CMakeLists.txt` extended. That is a build change:
  rebuild, re-run the tests and `verify --quick`, record the new `build_info()` (KNOWLEDGE.md 2.4).

## 3. Prepare the machine for timing

Accuracy can be measured anywhere; latency only on a quiet, dedicated machine (docs/BENCH.md "Usual
pitfalls" and "Keeping other GPU users away"; KNOWLEDGE.md 7.1):

- **No other GPU users.** `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`
  lists nothing but your run, before and after. Where allowed, `nvidia-smi -c EXCLUSIVE_PROCESS`; on
  a shared machine serialise runs with a lock file or the job scheduler; no desktop on that GPU.
- **Pinning.** The tuner neither pins itself nor checks load: run it under `taskset -c N`, N an
  isolated core (all its threads, `--workers` included, then share that core). `bench/bench.py` pins
  with `--cpu N` and refuses to time unpinned. On a dual-CCD V-Cache CPU use the V-Cache CCD.
- **CPU frequency.** Governor and energy-performance preference `performance`; bench.py records them,
  boost, hypervisor and host load, and warns when a condition is not met.
- **GPU clocks.** One clock profile for the whole job, stated. Clocks change latency, never outputs
  (KNOWLEDGE.md 4.5); run `verify --quick` again after changing the profile. An idle GPU drops its
  clock: at a 10 ms gap without keep-warm the call was +4.7 µs p50 on the overclocked RTX 5080 + Ryzen 7
  9850X3D setup and +2.5 µs at stock clocks with the final build
  (`tools/results/final-bench/final-bench.json`), and +6.3 µs p50 with the earlier build 6e on the overclocked setup
  (`tuning/data/knob_facts.json: call_cadence`) (KNOWLEDGE.md 2.3, 4.4).
- **Measure the machine's noise first**: `python bench/bench.py c-latency --quick --problem
  model-p1.dem --cpu N --out aa-check`. Its `warnings` list must be empty; its A/A control is the
  smallest difference the machine resolves. On the timing-grade, overclocked RTX 5080 + Ryzen 7 9850X3D
  setup the A/A band of `a03125c3` on bb144 xyz was 0.16 µs p50 / 0.22 µs p99 (0.5 % of the call;
  `tools/results/final-bench/final-bench.json`); on shared development machines the recorded A/A floors were
  p50 0.048-0.21 and p99 0.17-4.2 and steps below ~3 % were not readable
  (`tuning/data/knob_facts.json: dead_ends`; KNOWLEDGE.md 7.1, 6.6).

## 4. Measure the stock baseline and set the bar

"Stock" is ldpc BP+OSD-0 at its common settings: min-sum, alpha 0.625, 30 iterations, OSD-0. The
preset `ms30` is bit-identical to it, so its LER is stock's (KNOWLEDGE.md 5.1). Measure it at every
noise point:

```bash
python bench/bench.py ler --problem model-p1.dem --config ms30 --ler-shots 1000000 --out stock-p1
```

`stock-p1/results.json` holds `ler.shots` and `ler.variants.A.errors`. The whole set is sampled at
once (one byte per detector per shot): size `--ler-shots` to memory and to enough failures (the
paired-ratio rule of the project's measurement protocol, KNOWLEDGE.md 7.2, asks for >= 100
reference failures; the tuner itself does not enforce it). For H/O, use the API on i.i.d. errors:

```python
import numpy as np, eldpc; from scipy.stats import beta
rng = np.random.default_rng(seed)                 # a seed no other stage uses
fails = shots = 0
with eldpc.Decoder(H, O, error_rate=p, config="ms30") as dec:   # H, O: dense 0/1 uint8 arrays
    for _ in range(chunks):
        e = (rng.random((4096, H.shape[1])) < p).astype(np.uint8)
        obs = dec.decode_batch((e @ H.T) % 2)[1]
        fails += int(np.any(obs != (e @ O.T) % 2, axis=1).sum()); shots += len(e)
upper = beta.ppf(0.99, fails + 1, shots - fails)  # one-sided 99 % Clopper-Pearson bounds
lower = beta.ppf(0.01, fails, shots - fails + 1) if fails else 0.0
```

**The bar.** Acceptable means an LER cleanly below stock's at the same noise point: the one-sided 99 %
Clopper-Pearson upper bound is below a bar set with a clear margin under stock's LER, and stock's own
lower bound lies above that bar. On the reference problem (bb144 xyz, p = 1e-3) stock was 2.2e-4 and
the bar 1e-4 (KNOWLEDGE.md 5.1). Never set the bar above stock.

## 5. Run the tuner

The tuner is `tuning/eldpc_tune.py` (program name `eldpc-tune`); tuning/README.md is its manual.

**Quick run (smoke test)**: `python tuning/eldpc_tune.py --dem model-p1.dem --ler-target 0.05
--p99-budget-us 1000000 --auto --quick --out tune-quick` shows that the problem loads, the decoder
runs and the outputs are written ("seconds to minutes, depending on the problem", tuning/README.md).
Its 2,048 shots per point prove only loose targets (smallest provable bound about 4.6 / shots).

**Full run**, on the prepared machine, once per DEM file:

```bash
taskset -c N python tuning/eldpc_tune.py --dem model-p1.dem --ler-target BAR \
    --p99-budget-us US --cadence-us GAP --auto --out tune-p1
```

- Inputs: exactly one of `--dem`, `--circuit`, `--h FILE [--o FILE]`, `--problem NAME` (bb72, bb144,
  bb144xyz, bb288, hgp98, sc5, sc9, sc13, sc17; `--rounds N`; `--noise-model uniform|si1000` for the
  surface codes). `--p P1 P2 ...` applies only to `--problem` and `--h`: a point must then meet the
  one `--ler-target` at every rate, so when the bar differs per rate, run once per rate.
- Budgets: `--p50-budget-us` and/or `--p99-budget-us`, on the Python `decode_full` call. `--cadence-us`
  is the call gap (0 = back to back); above 500 µs keep-warm is evaluated.
- No hard bar: `--weight W` minimises log10(LER upper bound) + W * log10(p50); every point then runs to
  `--max-shots`, so set it. Without `--ler-target` only points whose screen LER is no worse than stock
  `ms30` on the same shots can be chosen (`--max-ler X` replaces that floor), and the choice must pass
  its own latency confirmation (below).
- `--seed N` (default 20261006) seeds every stage; `--max-shots N` (default 131072; below a bar of
  about 3.5e-5 a full run derives about 4.7 / bar); `--workers N` (parallel accuracy points, identical
  records). No duration is documented for a full run: read the quick run's `tool.duration_seconds`.

**Stages** (tuning/README.md "The procedure"): (1) a fixed coarse grid, alpha {0.25, 0.3125, 0.375,
0.5, 0.625, 0.75} x max_iter {1, 2, 3, 5, 7, 11, 15, 31} with BP+OSD-0 plus BP alone on a subset, each
point stopped by a sequential one-sided 99 % Clopper-Pearson rule; a larger cap of a passing
(decoder, alpha) row is skipped only when the passing point's non-converged share is below 1 - q at
the tuner's confidence (q = 0.99 under a p99 budget, 0.5 under a p50-only budget or none), and
`--no-prune` disables the skip; (2) refinement on fresh shots around the frontier (alpha ±1/32,
±1/16); (3) paired ABBA latency of each survivor against the current best with an A/A band, every
measurement on its own freshly sampled syndromes, the three fastest-predicted points at four times
the calls; (4) a fixed-size accuracy confirmation of the leading candidates on fresh shots, and a
latency confirmation of each; (5) the choice and, above 500 µs cadence, keep-warm off against on for
the chosen configuration; (6) on a GPU whose device class has two core variants, the chosen
configuration timed under both (below).

### Objective: fastest under an LER bar vs lowest LER under a latency budget

The tuner answers one of two questions; they are different runs.

- **Fastest configuration under an LER bar.** Give `--ler-target BAR` and a latency budget:

  ```bash
  taskset -c N python tuning/eldpc_tune.py --problem bb144 --p 0.003 --ler-target 1e-2 \
      --p99-budget-us 1000 --auto --out tune-bar
  ```

  The screen stages stop each point as soon as a bound decides it (cheap), the latency stage
  measures the survivors, the survivors near the budget are re-timed before the in/out decision
  (**Budget membership** below, the same rule as the second recipe), and the accuracy
  confirmation re-measures the in-budget candidates on a fixed number of fresh shots, one at a
  time in the choice's order, until one is accepted (**Confirmed choice** below). The reported LER
  is the confirmation's; the screen's number is shown next to it, labelled a screen. A candidate is
  accepted only if its confirmation's upper bound is below BAR at every noise point and its latency
  confirmation is within the budget; the first accepted candidate in the choice's order wins. If
  none is accepted there is no choice: the run says so, lists the candidates with their screen
  numbers (and why each failed) and the nearest points, writes
  `config.nearest.json` and no `config.json`, and exits with status 3 (0 = a choice written, 3 = no
  configuration qualifies, any other non-zero = an error; tuning/README.md "Output files").
- **Lowest LER under a latency budget.** Give `--weight 0 --p99-budget-us B` and no `--ler-target`:

  ```bash
  taskset -c N python tuning/eldpc_tune.py --problem bb72 --p 0.001 --weight 0 \
      --p99-budget-us 1000 --max-shots 2048 --auto --out tune-lowest-ler
  ```

  Every point is measured on `--max-shots` shots per case (there is no bar to stop a point
  early). Four rules decide the choice:
  - **LER floor.** A point is choosable only if, at every noise point, its screen LER is no worse
    than stock `ms30` (ldpc's common settings) decoded on the same screen shots; the screen decodes
    `ms30` as a reference for that, not as a candidate. BP alone and every point worse than stock
    are therefore never chosen, and they are not latency-measured. `--max-ler X` replaces the floor
    by "screen LER at most X" (`--max-ler 1` disables it). `results.json` "ler_floor" and the
    `pareto.md` section "LER floor" record the floor, its source and the excluded points.
  - **Budget membership** (both recipes since tuner 2.2). A choosable point (with `--ler-target`, a
    screen pass) whose latency-stage p99 (or p50, for a p50 budget) is within 0.5x-2x of the
    budget is re-timed before the in/out decision: at least 8000 calls per
    side on fresh syndromes, paired with the A/A reference, as the latency confirmation does; that
    measurement decides. The margin covers the scatter seen where the p99 sits on a ~1 % host tail
    or a bimodal OSD tail (RTX 6000 Ada with a two-socket EPYC 7542, development record): the
    stage's pooled 2000-call p99 read down to 0.64x a point's 8000-call p99 (88.4 against
    137.9 µs), its 1000-call runs 0.53-1.9x the 4000-call runs, and two runs of one OSD-tail point
    differed 2x (345 and 708 µs).
    At most 32 points are re-timed, nearest to the budget first (a few seconds each); the rest keep
    their latency-stage verdict and are listed as capped. `results.json` "budget_retime" and the
    section "Budget membership" have every re-timing and its wall time.
  - **Confirmed choice** (both recipes). The in-budget points are ranked (lowest LER upper bound
    first at `--weight 0`; fastest p50 first under a bar) and confirmed one at a time in that
    order until one is accepted -- its own latency confirmation is within the budget (under a
    bar, also its accuracy confirmation's bound is below BAR) -- or every in-budget point has been
    confirmed; the first accepted one is the choice. If none is,
    there is no choice: the run says "no configuration qualifies", lists the nearest points, writes
    no `config.json` (only `config.nearest.json`) and exits with status 3, as the bar recipe does.
    Each nearest line prints the p50 / p99 of the measurement its budget ratio comes from and names
    it (latency confirmation, re-timing at N calls, or the latency stage).
  - **The LER claim** (`choice.ler`, `ler_lower`, `ler_upper`, `ler_source: "screen"`) is the
    screen's fixed-size estimate -- 131,072 shots per case by default, unbiased because no point
    stops early -- with its one-sided 99 % Clopper-Pearson bounds, worst case over the noise
    points. The small accuracy confirmation (2048 fresh shots) is reported as a consistency check
    (`choice.confirmation_check`); the tool warns when it disagrees with the screen.

**Cost.** With `--weight` and no target every point runs to `--max-shots`, so the accuracy time
grows with the grid size times `--max-shots`; a target run stops each point as soon as a bound
decides it and is usually much cheaper. One public problem: the second command above (built-in
bb72, full grid, not `--quick`), run with `--out tuning/results/bb72-weight0` on an RTX 5080 with a
Ryzen 9 7950X3D with tuner 2.0 (before the floor, the re-timing and the confirmed-choice rule, which
add one `ms30` screen point per stage and a few seconds per re-timed point), took about 21 s -- `results.json` there records `tool.duration_seconds` = 20.892
and the exact command, and its `RUN.md` repeats both with the machine. A single run on one
machine; a surface or qLDPC code is slower per shot, so time your own problem rather than scaling
this number.

**Confirmation details.** The shot count N is declared before the first confirmation shot and
recorded in `accuracy_confirmation.shots_rule`. N doubles from 64 until a point whose true LER is
`target / 1.5` passes (one-sided upper bound below the target) with probability at least 0.9, and a
bisection below that N returns an N at which the probability is at least 0.9 and N - 1's is below
it. The pass probability is not monotone in N, so N is not necessarily the smallest count that
meets the margin (target 0.05 at confidence 0.99: N = 1944, while 1899 also does).
`--confirm-shots N` overrides it; `--quick` caps it at 2048. The candidates are the screen
survivors within the latency budget after the re-timing, in the choice's order: the screen
choice's point first, then the other in-budget points (fastest p50 first under a bar; the ranking
without one). A point over the budget is never confirmed. They are confirmed one at a time
(accuracy, then latency) until one is accepted or the in-budget list is exhausted. Tuner 2.1
stopped after a fixed three, so a run could say "no configuration qualifies" while in-budget
screen passes that would have qualified were never confirmed. There is no cap by default; each extra candidate
costs one accuracy confirmation (N shots per case) and one latency confirmation (8000 calls), and
the list is at most the in-budget part of the grid. `--confirm-candidates K` caps the count; when
the cap stops the confirmation with in-budget candidates left, the run warns, names them
(`accuracy_confirmation.unconfirmed`, `stopped: "cap"`) and, with no accepted candidate, exits
with status 3 although one of them may qualify. Every confirmation is a one-sided test at the
tuner's confidence, so confirming k candidates whose true LER sits at the bar lets one through
with probability up to 1 - 0.99^k; such a point also had to pass the screen's own bound, and
`accuracy_confirmation` keeps every confirmed candidate (`confirmed`, `failure_reasons`) to read
it. Every candidate decodes the same N fresh shots of each case (their own seed stream, distinct
from every screen, latency and keep-warm stage), and no candidate stops early. The report shows,
per candidate, the screen's upper bound next to the confirmation's failures and bound, and why a
candidate was not accepted.

**Pruning.** Under a latency budget at quantile q (p99 budget: q = 0.99; p50 only: q = 0.5; no
latency budget: q = 0.5) the larger caps of a (decoder, alpha) row may be skipped only when the
passing point's **converged share** is high enough: the one-sided upper bound of its non-converged
share at the tuner's confidence must be below 1 - q. While at least 1 - q of the calls take the
slow path (BP did not converge: OSD-0 for BP+OSD, the full cap for BP), the q quantile sits inside
the slow class, and a larger cap that moves shots from OSD-0 into the converged class can be
faster there -- so such a pass does not dominate. A skipped point keeps `dominated_by` and records
`dominator_converged_share`. `results.json` "pruning" records the threshold (`min_converged`), the
quantile, the confidence and the rule. On an OSD-bound problem (many shots on the slow path, a low
converged share) expect little or no pruning and read the converged shares before trusting a cap
comparison; `--no-prune` evaluates every point of every row.

**Core variants.** On a GPU whose device class has two core variants (compute capability 8.9 with
at least 128 SMs, where the package picks `ada64`; docs/BUILD.md "Core variants") the faster variant
depends on the problem and the preset: at long caps on bb144 xyz `ada64` is x1.09-1.37 slower,
while at the default preset it is faster on every measured problem and at long caps the other
measured problems stay level or faster. After the choice the tuner
therefore times the chosen configuration under both variants: one handle of each (it sets
`ELDPC_DEVICE_CLASS` around each create in its own process and restores the previous value), paired
in the ABBA schedule on the same fresh syndromes at the top candidates' call count, and an A/A of two
handles of the table's variant for the band. A variant is reported faster only when its p50 and p99
are both beyond the A/A band (worst case over the noise points); otherwise "no difference". Both
variants run the same decoder sources, so there is no exactness check. `results.json`
"core_variants" and the `pareto.md` section "Core variants" record both measurements, the ratios,
the band, `faster` and the recommendation; the console prints it. When `default` wins, set
`ELDPC_DEVICE_CLASS=default` in the environment of the process that creates the decoder (the
recommendation names it as `recommended_env`); `config.json` does not carry the variant, because
`Decoder(config=...)` has no such key and the front reads the variable from the process environment
at every create. The other latency numbers of the run were taken with the table's variant (or with
the variable you set for the run, recorded as `environment_value`); each latency confirmation
records `core_variant`. On a one-variant device the stage is skipped and records why.

**Resume** an interrupted run with the identical command plus `--resume`: screen accuracy progress
in `DIR/raw/checkpoint.json` is reused, and the fixed-shot accuracy confirmation continues from
`DIR/raw/confirmation-checkpoint.json` -- its progress is saved after every decoded chunk even
without `--resume`, so an interrupted run leaves it behind, and a resumed candidate whose saved
count is the full N is not decoded again.  Latency is confirmed again.  Other inputs, another
grid, another pruning rule, another confirmation shot count or another tuner version are refused.

## 6. Read the outputs

- `config.json`: the choice; pass `config="config.json"` to `eldpc.Decoder` or `Decoder.from_dem`.
  If nothing qualifies there is `config.nearest.json` instead, which does not qualify, and the
  exit status is 3.
- `pareto.md`: environment, choice, and one row per point: LER per case with its 99 % interval, p50,
  p99, in budget, Pareto star, note (`CHOSEN` on the accepted choice, "screen pick (not accepted)"
  on a passed-over screen pick, dominated, latency verdict).
- `results.json`: every number. Check `environment` (the GPU, driver and build you meant; the
  CUDA `toolkit` is the loaded library's, read from its `build_info()` with `source:
  "build_info"`, or from `nvcc --version` with `source: "nvcc"` only when the library names none),
  `latency.aa` (`band_p50`, `band_p99`, fractions), `latency.experiments` (verdicts `faster`,
  `within_noise`, `slower`; each with its `distinct_syndromes` and syndrome seeds),
  `accuracy_confirmation` (`planned_shots`, `shots_rule`, `cases` with their seeds, per candidate
  `screen`, `per_p`, `passed`, `latency` -- its latency confirmation with `distinct_syndromes`,
  `syndrome_seeds`, the slow-class fields and `core_variant`, for every candidate, chosen or not --
  `latency_within`, `accepted`, `failure_reasons`, `resumed_case_shots`; and `chosen`,
  `in_budget_candidates`, `in_budget_order`, `confirmed`, `stopped`, `unconfirmed`,
  `max_candidates`),
  `pruning` (`min_converged`,
  `quantile`, `confidence`, `rule`), `choice` (`config`, `ler` and `ler_upper` from the confirmation,
  `ler_source`, `screen`, `p50_us`, `p99_us`, `slow_fraction`, `p99_unreliable`, `tie_set` with
  `chosen` on the accepted choice and `screen_pick` on the screen's pick, `screen_choice_label`,
  `reason`; with no choice `nearest`, each entry's `p50_us` / `p99_us` / `budget_distance` from
  the one measurement it names in `measurement`), `confirmation` (the latency confirmation of the final choice), `keep_warm`,
  `core_variants` (the variant comparison, or why it was skipped), `warnings` (one per reported
  candidate whose p99 is unreliable, the capped re-timings, a `--confirm-candidates` cap that left
  candidates unconfirmed, and without `--ler-target` a passed-over candidate and an inconsistent
  confirmation), `budget_retime` (both recipes: the re-timing margin, every re-timed point and its
  wall time) and `ler_floor` (runs without `--ler-target`: the floor with its `ms30` references
  and excluded points),
  `problem_cases` (mechanism count, `dem_options`, DEM sha256) and `stages` (per point: `pass`,
  `fail`, `inconclusive` or `skipped`, `converged_share`, chunk history, seeds).
- **Screen vs confirmation.** The screen stages are cheap because they stop at the first favourable
  look; their per-point LER estimates are biased low and are not the number to report. The
  confirmation has a fixed shot count and no early stop. In `pareto.md` the pareto table keeps the
  screen numbers (the table is the map of the search); the Choice and Accuracy confirmation sections
  give the confirmation's LER and upper bound, and the screen's number next to them labelled as the
  screen. Without `--ler-target` no point stops early, so the screen's fixed-size number is the
  claim (`ler_source: "screen"`) and the 2048-shot confirmation is only a consistency check: in the
  tuner re-validation (development record) that confirmation read 0.55x and 4.25x the fresh 1M-shot
  LER, the 131,072-shot screen 0.88-1.00x.
- **Slow class.** `slow_fraction` is the share of timed calls whose status word lacks the converged
  flag. `p99_unreliable` is true exactly when `slow_fraction` lies in [0.005, 0.02]: at about 1 %
  slow calls the p99 is one of them and moves with how many fell in the sample. Then the p99 budget
  verdict uses the median latency of the slow calls (`slow_median_us`), and the tool warns -- for
  every reported candidate where it is set: each confirmed candidate's latency confirmation and
  each measured row of the Pareto table (`results.json` "warnings", the `pareto.md` section
  "Warnings" and the console). Read `slow_fraction` before quoting any p99.
- **The tie set**: candidates whose p50 (and, under a p99 budget, p99) is within the A/A band of the
  fastest. The winner has the lowest LER upper bound, then fewer iterations, then the alpha closest to
  a shipped preset, then the label. Report the whole set.
- Tuner latencies: Python call, interpolated quantiles; bench.py: C call, nearest-rank. Name the method.
- Keep-warm is application-side, not in `config.json`. If recommended, call `Decoder.keep_warm()`
  (C: `eldpc_keep_warm(h, ELDPC_KEEP_WARM_GPU)`) every <= 0.5 ms while idle, the last call shortly
  before the decode, never concurrently with a decode on that handle (KNOWLEDGE.md 2.3).

## 7. Decide

1. **Acceptable**: the one-sided upper bound of the confirmation's LER is below the bar at every
   noise point the deployment must cover. Screen passes are not enough; inconclusive points do not
   qualify. `results.json` "choice" is null when no confirmed candidate qualifies (exit status 3).
   Without a bar (`--weight`), the choice is above the LER floor and inside the budget by its own
   latency confirmation, and its LER is the screen's fixed-size estimate; a null choice
   means no such point exists on this machine at this budget.
2. **Take the fastest acceptable configuration** on the deployment machine, at the budget's quantile
   and API; differences within the A/A band are ties, never wins. Accept a lossier one only for a
   worthwhile speed gain; show the trade as a Pareto table (latency against LER per noise point).
   The choice's p50/p99 are from its latency confirmation, not from the screen's paired runs.
3. **Read the pruning rule and the converged share.** A skipped point is only as safe as the
   dominator's converged share: if the passing point had a low converged share (many shots on the
   OSD path), the larger caps of its row were evaluated, and the skipped points have
   `dominator_converged_share` recorded. On an OSD-bound problem -- a low converged share and many
   shots on the slow path -- do not read a p99 comparison of adjacent caps from the screen alone;
   widen the search with `--no-prune`, or compare the candidate configs directly with the benchmark.
4. **Read slow_fraction and p99_unreliable before quoting a tail.** A p99 whose slow-class share is
   within 2x of 1 % is not a stable number; the tool marks it (`p99_unreliable`), warns, and for a
   p99 budget falls back to the median of the slow calls. If the slow class is large, a p99 budget
   and a p50 budget answer different questions -- quote both, and compare configurations paired in
   one `bench.py` run (`--config A --config B`), never in sequential runs on a shared machine.
   `bench.py`'s `gpu_processes_between` record lists other compute processes it saw between
   measurements (a lower bound: a process that starts and ends inside one measurement is missed),
   and its `warnings` include a shot far above the p50 (more than 50x and above 1 ms). Either means
   the tail is not the decoder's; repeat the run on a quiet machine.
5. **Check more than one noise point.** On bb144 xyz `a03125c3` is 0.591x the (0.5, 31) reference's
   LER at p = 1e-3 but 1.066-1.074x at 3e-3; on bb144 z `a04375c3` is about 10x below stock at 1e-3,
   while low alpha at cap 7 is 1.5-3.5x worse than the reference at 2e-3 (KNOWLEDGE.md 3.4, 5.4, 5.5).
   Take a configuration that qualifies in the run of every noise point.
6. **Bimodal configurations** (many shots converged and many on the OSD path): judge per class, not on
   the pooled p50 (KNOWLEDGE.md 3.2); bench.py keeps per-shot times and status words in `DIR/data/`.
7. **Confirm** on fresh shots and in C. bench.py's LER shots depend on the file name and the shot
   count, so use a count M other than the stock run's; the `ms30` side is then stock on the same shots:

   ```bash
   python bench/bench.py ler --problem model-p1.dem --config ms30 --config tune-p1/config.json \
       --ler-shots M --out confirm-ler-p1
   python bench/bench.py c-latency --problem model-p1.dem --config a03125c3 \
       --config tune-p1/config.json --cpu N --gap-us GAP --out confirm-c-p1
   ```

   For a C budget, `taskset -c N python tools/measure_decode_overhead.py --dem model-p1.dem --config
   tune-p1/config.json` measures the Python layer inside the tuner's numbers.

**Widen the search** when only cap 31 qualifies, the winner's alpha is the lowest or highest the run
evaluated, nothing qualifies, the problem is OSD-bound (a low converged share: the larger caps were
not pruned, but the grid may still stop too low), or KNOWLEDGE.md names a frontier point outside the
grid (cap 9 is not in it: (0.375, 9), section 5.3). Pass extra points to bench.py `ler` (paired with `ms30`) and `c-latency`
(paired, A/A control) as quoted ABI strings, keys as in `eldpc.presets()`, and judge them by the rules
above. Do not edit `tuning/eldpctune/grid.py` for one job: its tests pin the grid and a resume refuses
another grid. If nothing qualifies, report the nearest points; do not loosen the bar.

**Never change** `src/core/` (pinned by `SOURCES.sha256`; code identical to the benched sources, `TOKENS.sha256`), `src/third_party/`, or the exactness
flags in `CMakeLists.txt` (host `-ffp-contract=off -fno-fast-math`, no `-march=native`; nvcc
`-fmad=false -ftz=false -prec-div=true -prec-sqrt=true`). A change that alters any output (correction,
observables, status word, soft output) is not a tuning result, however fast. Build-time knobs
(KNOWLEDGE.md 2.4) are builds: rebuild, re-run the tests and `verify --quick`, record `build_info()`.

## 8. Use the knowledge base

| question | KNOWLEDGE.md |
|---|---|
| exactness classes; what transfers between machines; the workflow; starting configs by code family | 0.1, 0.2, 0.4 |
| what one call does and where its time goes; the OSD path | 1 |
| config keys, presets, platform knobs (keep-warm, cadence, pinning, clocks, ISA), build knobs, create-time shape | 2 |
| alpha with the cap, the converged share, code size, noise level | 3 |
| latency against the cap, LER surfaces, keep-warm, clocks, SM/L2 budgets, toolkit, host ISA | 4 |
| acceptance rules, the stock baseline, Pareto tables and frontiers | 5 |
| ideas measured as losses or inexact | 6 |
| measurement method and the reporting checklist | 7 |
| what is not measured (among it absolute latency on the RTX 6000 Ada board itself; its Sapphire Rapids-W host family is measured with an L40S, machine N) | 8 |

Do not repeat a dead end of section 6 without a new reason and a measurement; platform-scoped ones
(6.5) may differ on another GPU, driver or host, so name that difference. **Adding a result.** Follow KNOWLEDGE.md 7.3 (machine, build, cadence, runs, shots, latency, LER,
exactness, problem). Every number comes from a results file of the job; cite file and field (for
example `tune-p1/results.json: choice.p50_us`). Name the hardware with a new key in the 0.2 table and
its grade (timing-grade only for a dedicated, pinned, quiet setup with an empty bench.py `warnings`
list). Mark arithmetic as computed. Where a number disagrees with a row, keep both with the condition
that separates them. A new dead end goes into section 6 with the number that decided it, its machine
and build; a closed gap leaves section 8.

## 9. Deliverables of a tuning job

- the chosen `config.json` (or `config.nearest.json`, stated as not qualifying);
- per noise point the tuner's `results.json`, `pareto.md` and `raw/`; the bench.py directories
  (`aa-check`, stock LER, confirmations); `verify-quick/summary.md`;
- a short report: hardware and environment (section 1, `build_info()`, clock profile, pinned core),
  the A/A bands (tuner `latency.aa`, bench.py), per noise point the confirmation's LER with its
  planned shots and 99 % interval next to stock's LER (and the screen's number labelled as the
  screen), the converged shares and the pruning decisions, the choice and its tie set, p50 and p99
  with API, quantile method and cadence, the slow-class shares, the keep-warm decision, and every
  command.

## 10. Common mistakes

- **Single-run timings.** Use paired runs with an A/A band (the tuner and bench.py do; by hand, at
  least 3 main and 3 A/A processes, KNOWLEDGE.md 7.1).
- **Tuning on a shared machine** or unpinned: the LER stays valid, the latency does not.
- **Comparing across sessions, GPUs or machines.** Compare within one paired session; across machines
  use `python bench/bench.py compare --a DIR_A --b DIR_B` and quote both A/A bands.
- **Reused seeds.** The tuner's default `--seed` repeats the same shots; bench.py repeats them for the
  same file name and shot count. A confirmation needs shots no earlier stage used -- the tuner derives
  its confirmation seed stream from `--seed` with its own stage name, so do not rerun a different
  confirmation by editing the seed of a stage by hand.
- **Quoting the screen's LER.** A screen point stops at a favourable look; its estimate is biased
  low. Report `choice.ler` (the confirmation's), and label the screen number when you show it.
- **Quoting p50 when the budget is p99**, or the Python number for a C budget.
- **A CSS half passed to `--h` without `--o`**: the derived observables include stabilizers; LER wrong.
- **Mixing built-in names.** bench.py's `bb144` is X and Z jointly and its `surface` uses uniform noise;
  the tuner's `bb144` is Z memory and its `sc*` default to SI1000. Use one DEM file everywhere.
