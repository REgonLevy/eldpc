# eldpc-tune: the knob tuner

`eldpc-tune` finds the fastest decoder configuration that meets a logical error rate target and a
latency budget for one code and its noise, and writes a config file that `eldpc.Decoder(config=...)`
loads.  The knobs it searches are the min-sum scaling factor (`ms_scaling_factor`, alpha), the BP
iteration cap (`max_iter`), the decoder (BP+OSD-0, or BP alone where the target allows it), and
keep-warm on/off for sparse call cadence.

The procedure is fixed and deterministic for the accuracy stages: the same `--seed` gives the same
shots, the same failures and the same accuracy choices.  Latency is a measurement of the machine
and is not bit-reproducible.

## Requirements and running it

The tuner needs the installed `eldpc` package (with a working CUDA device), numpy, scipy and stim.
The built-in `--problem` recipes build their circuits with the optional `qldpc` package (which
brings `sympy`; `pip install ".[qldpc]"` from the repository root installs the tested revision);
`--dem`, `--circuit` and `--h` do not need it.  From the repository root:

```sh
python tuning/eldpc_tune.py --problem bb144xyz --p 0.001 \
    --ler-target 1e-4 --p99-budget-us 60 --auto --out tune-bb144xyz
```

A smoke test with small shot counts and a smaller grid (seconds to minutes, depending on the
problem):

```sh
python tuning/eldpc_tune.py --problem bb72 --p 0.001 --rounds 6 \
    --p99-budget-us 1000000 --ler-target 0.05 --auto --quick --out tune-quick
```

`--quick` decodes 2,048 accuracy shots per point, so it can certify only loose LER targets: the
smallest upper bound 2,048 shots can prove is about 4.6/2,048 = 2.2e-3 (see "Reading the outputs").
With `--ler-target 1e-3 --quick` no point can qualify; the tool warns, says "No configuration
qualifies", writes `config.nearest.json` and exits with status 3 (see "Output files").  Use `--quick` for smoke tests like the one above, and
a full run for a real target.

Without qLDPC the same kind of run works on a stim detector error model:

```sh
python tuning/eldpc_tune.py --dem my_code.dem --ler-target 1e-3 --p99-budget-us 100 \
    --auto --out tune-dem
```

A DEM file can be written from any stim circuit, for example a surface code without qLDPC:

```sh
python -c "import stim; c = stim.Circuit.generated('surface_code:rotated_memory_z', distance=5, rounds=5, after_clifford_depolarization=1e-3); open('my_code.dem', 'w').write(str(c.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)))"
```

The README section "From a code to a circuit, a DEM and a decode" shows the same for a qLDPC code,
and every `bench/bench.py` run writes the DEM of its problem to `DIR/data/model.dem`.

Without `--auto` the tool prints the plan and asks once for confirmation when it runs on a
terminal; it never prompts otherwise.  There is no language model anywhere in the procedure.

### Problem inputs (give exactly one)

| option | meaning |
|---|---|
| `--problem NAME` | built-in recipe: `bb72`, `bb144` (Z memory), `bb144xyz` (X and Z jointly), `bb288`, `hgp98`, `sc5`, `sc9`, `sc13`, `sc17` (rotated surface codes, SI1000 noise) |
| `--dem FILE` | stim detector error model; shots are sampled from the DEM |
| `--circuit FILE` | stim circuit; the DEM is derived undecomposed (`decompose_errors=False`, `approximate_disjoint_errors=True`) unless `--decompose` is given, and shots come from the circuit |
| `--h FILE` | parity-check matrix for a code-capacity problem (text `m n` then m rows of 0/1, or the binary `DBCSC001` format: the 8 bytes `DBCSC001`, nrows, ncols and nnz as little-endian uint64, then ncols+1 uint32 column pointers and nnz uint32 row indices); `--o FILE` supplies the observable matrix (k rows, same columns as H); read the warning below before leaving it out |

`--noise-model uniform|si1000` overrides the built-in recipe's noise model: surface codes default
to SI1000 and also accept uniform depolarizing noise; bb and hgp problems always use uniform noise
(an override there is refused).  `--p` sets the physical error rate (one or more; with several rates
a point must meet the target at every rate).  It defaults to 1e-3 for `--problem` and is required
for `--h`.  `--rounds` overrides the round count of a built-in recipe.  The built-in recipes are
the standard memory experiments: bivariate-bicycle and hypergraph-product codes through qLDPC's
memory experiment with uniform depolarizing noise, surface codes through stim's generated circuit
with SI1000 noise.

The built-in recipes and `--circuit` derive the DEM undecomposed by default
(`decompose_errors=False`, `approximate_disjoint_errors=True`), which is the model BP+OSD decodes:
its error mechanisms are the DEM's error instructions.  `--decompose` adds
`decompose_errors=True` and `ignore_decomposition_failures=True`, and `--no-decompose` selects the
default; the two produce different DEMs (different mechanism lists and `dem_sha256`) and therefore
different problems.  `--dem FILE` takes a DEM as written.  Every entry of `results.json`
"problem_cases" records the `dem_options` passed to stim next to the mechanism count.

**Warning for `--h` without `--o`.**  The observables are then derived as a basis of
`ker(H)/row(H)`, which is the logical operator group only when `H` is the *full* commuting
stabilizer matrix of the code (`H H^T = 0` over GF(2) is checked, but it is necessary, not
sufficient).  It is not a CSS logical basis.  If `H` is one CSS half (for example `Hx` with
`Hx Hx^T = 0`), the derived rows include the stabilizers of the other type, a decoded stabilizer
component is counted as a logical error, and the LER is wrong.  For a CSS half, pass the observable
matrix explicitly with `--o`.  The tool prints a warning whenever it derives the observables.

### Targets and budgets

- `--ler-target P`: the logical error rate per shot must be below `P`, judged by a one-sided 99%
  Clopper-Pearson upper bound.
- `--p50-budget-us US` / `--p99-budget-us US`: the measured single-shot call latency must be within
  the budget (both when both are given).
- `--weight W`: without a hard target (or among the points that meet it), choose the point with the
  smallest `log10(LER upper bound) + W * log10(p50_us)`.  `W = 0` favours accuracy, larger `W`
  favours latency; for example `W = 1` accepts a 10x latency increase for a 10x lower LER.  Without
  `--ler-target` every grid point is measured to `--max-shots`, so set a shot budget (or use
  `--quick`) for a cheap weight-only run.  Without `--ler-target` (with or without `--weight`) a
  point is choosable only above the **LER floor**, the choice must pass its own **latency
  confirmation**, and the reported LER is the screen's fixed-size estimate (steps 2, 5 and 6
  below).  In both modes budget membership near the budget is decided by a **re-timing** (step 4),
  and when nothing qualifies the run writes no `config.json` and exits with status 3.
- `--max-ler X` (only without `--ler-target`): replace the default LER floor -- stock `ms30`'s
  screen LER on the same shots -- by X: a point is choosable only when its screen LER is at most X
  at every case.  It caps the screen's estimate and proves nothing (`--ler-target` proves a
  bound); `--max-ler 1` disables the floor.
- `--cadence-us US`: the application's call spacing (0 = back to back).  At non-zero cadence the
  latency loop waits between calls and the keep-warm knob is evaluated.
- `--confirm-shots N`: the fixed number of fresh shots per candidate and case in the accuracy
  confirmation.  By default it is derived from the target, the confidence and a planned margin
  (N doubles from 64 until a point whose true LER is `target / 1.5` passes with probability at
  least 0.9, then a bisection returns an N that meets it while N - 1 does not; the probability is
  not monotone in N, so this N is not necessarily the smallest that meets it); `--quick` caps it
  at 2048.
- `--confirm-candidates K` (default: no cap): an optional cap on the confirmation.  Without it the
  in-budget candidates are confirmed one at a time, in the order the choice ranks them, until one
  is accepted (it is the choice) or every in-budget candidate has been confirmed ("Accuracy
  confirmation" below).  With a cap, a run that stops at K candidates with in-budget candidates
  left unconfirmed warns, names them (`accuracy_confirmation.unconfirmed`) and says so in its
  reason; with no accepted candidate it exits with status 3 although an unconfirmed one may
  qualify.
- `--no-prune`: evaluate every point of every row; without it a larger cap of a (decoder, alpha) row
  is skipped only under the dominance rule described in "The procedure".
- `--seed N`, `--out DIR`, `--auto`, `--quick`, and the advanced `--max-shots N` (accuracy shots per
  point per case; default 131072, 2048 with `--quick`, and for a full run with a target below the
  default's zero-failure floor of about 3.5e-5 the default is derived from the target as about
  4.7/target) and `--workers N` (evaluate N grid points
  concurrently in the accuracy stages; default 1).  With workers above 1 the shot records and
  accuracy decisions are identical to a sequential run (the tests check this): each point still
  decodes the same shared chunks, so only wall time changes.  Handle creation is mostly host-side
  and overlaps; decoding small batches may not saturate the GPU while high-cap batches do, so the
  gain depends on the problem and the machine (a short `--quick` run with `--workers 1` and
  `--workers 4` shows it).  A group never holds two points
  of the same (decoder, alpha) row, so dominance pruning is decided against earlier groups exactly
  as in a sequential run and the same points are skipped.  The latency stage also creates the
  surviving points' handles concurrently (up to 32 handles; above that it creates them lazily).
- `--resume` continues an interrupted run from `DIR/raw/checkpoint.json`, and the fixed-shot
  accuracy confirmation from `DIR/raw/confirmation-checkpoint.json`.  A run writes the stage file
  after every evaluated group and the confirmation file after every decoded chunk (even without
  `--resume`, so an interrupted run leaves it behind); a resume reuses the completed points (their
  records are deterministic) and the confirmation shots already decoded (up to the last complete
  chunk: a partially decoded chunk is decoded again from its start, and a candidate/case whose
  saved count is the full N is not decoded at all).  Both files carry a fingerprint of the
  problem, seed and shot settings, the confirmation shot count, the tuner version, the pruning
  rule and a hash of the grid itself (the coarse point list and the constants of the refinement
  rule); every stage also records the hash of its own point list.  A resume with different inputs,
  another grid, another pruning rule, another confirmation shot count or another tuner version is
  refused.  A resumed stage reports the shots its recorded points decoded (`shots_sampled`, with
  `shots_sampled_this_run` and `resumed_points` alongside); `sample_seconds` accumulates over the
  interrupted runs.  Latency is always measured again, because it describes the machine at the
  time of the run.

## The procedure

1. **Fixed grid.**  A coarse grid over (alpha, max_iter) and the decoder: alpha in
   {0.25, 0.3125, 0.375, 0.5, 0.625, 0.75}, max_iter in {1, 2, 3, 5, 7, 11, 15, 31}, BP+OSD-0 at
   every point and BP alone at alpha {0.3125, 0.5, 0.625} and max_iter {3, 7, 15, 31}.  The ranges
   span the presets the package ships (`eldpc.presets()`): alpha 0.25-0.5 is the tuned side and
   0.625-0.75 the stock/high side, caps 2-3 are typically the smallest useful caps and 31 is the
   reference point.  On the bb144 circuit-level problems the grid was designed around, cap 1 failed
   the accuracy targets at every alpha tried and cap 2 was marginal; that is a property of those
   problems, not of the tool, and other problems (for example surface codes, where max_iter 1 can
   pass) behave differently, which is why cap 1 stays in the grid.  The grid is the same for every
   run; `--quick` uses a smaller fixed grid.
2. **Accuracy per point.**  Every point decodes the same freshly sampled shots in a stage (each
   stage derives its own seed from `--seed`; seeds are never reused across stages).  A point stops
   as soon as a one-sided 99% Clopper-Pearson bound decides it against the target: the upper bound
   below the target passes it, the lower bound above the target rejects it.  The bound is evaluated
   after every chunk; this is the prescribed sequential rule (the interval is not re-spent across
   looks).  Points that reach the shot limit undecided are reported as inconclusive and do not
   qualify.  Every evaluated point records its `converged_share`: the share of its screen shots
   whose status word has the converged flag (`eldpc.status`), per case in `per_p` and as the
   minimum over cases on the point.  **Dominance pruning.**  A point with the same decoder and
   alpha and a smaller cap that already passed may make a larger cap irrelevant, but only under a
   measured condition: the passing point's non-converged share must be below `1 - q` at the
   tuner's confidence (`--no-prune` disables pruning), where q is the latency quantile the budget
   protects (p99 budget: 0.99; p50-only budget or no latency budget: 0.5).  While at least
   `1 - q` of the calls take the slow path (BP did not converge: OSD-0 for BP+OSD, the full cap
   for BP), the q quantile sits inside the slow class, and a larger cap that moves shots from
   OSD-0 into the converged class can be faster there -- so a pass with a low converged share
   does not dominate.  A skipped point keeps `dominated_by` and gains
   `dominator_converged_share`; `results.json` "pruning" records `min_converged` (the q used), the
   quantile, the confidence and the rule.  On an OSD-bound problem most passes have a low
   converged share, so little is pruned and the larger caps of a row are evaluated.
   **LER floor** (runs without `--ler-target`).  Each screen stage also decodes stock `ms30`
   (alpha 0.625, 30 iterations, OSD-0: ldpc's common settings) on its own shots, as a reference,
   not a candidate (`stage.references`).  A point is choosable only when at every case its failures
   on those shots are no more than `ms30`'s (a paired, exact comparison); every other point -- BP
   alone in particular -- is recorded with `ler_floor.passed: false`, is not latency-measured and
   cannot be chosen.  `--max-ler X` replaces the floor by "screen LER at most X".  `results.json`
   "ler_floor" records the source, the rule, the references per stage and case, and the choosable
   and excluded points; `pareto.md` has a section "LER floor".
3. **Refinement.**  A second stage with fresh shots refines around the accuracy frontier:
   alpha neighbours (±1/32, ±1/16) of the good alpha values, evaluated at the smallest passing cap
   and at the cap just below it.  If nothing passes, the two caps with the smallest upper bounds
   are refined the same way.  At most 16 new points per decoder kind.
4. **Latency per surviving point.**  Every point that passed is measured with single-shot calls
   through the public package API (`Decoder.decode_full`) on freshly sampled syndromes, at the
   requested cadence.  Every paired measurement has its own seeded syndrome set: both handles of a
   measurement decode exactly the same shots (the pairing stays exact), and no two measurements
   share a shot.  Each measurement records `distinct_syndromes` and its syndrome seeds.  Points are
   measured in order of their mean decode time from the accuracy stage and each is paired against
   the current best in an ABBA schedule (reference, candidate, candidate, reference).  An A/A
   experiment (two handles of the same configuration through the same schedule, repeated twice)
   gives the band of paired ratios the machine produces when nothing differs; a candidate replaces
   the best only when both p50 and p99 improve beyond that band.  The three fastest-predicted
   survivors are measured with four times the calls, because the A/A band of the pooled p99 shrinks
   roughly with the square root of the call count; the rest use the base count with their own A/A
   band.  With a non-zero cadence the call count is scaled so one comparison stays a few seconds
   (at very sparse cadences the base and large counts coincide).  Every measurement also splits
   its timed calls by the status word: `slow_fraction` (the share without the converged flag),
   `p99_unreliable` (true exactly when `slow_fraction` is in [0.005, 0.02]) and the median latency
   of the slow calls.  Under a p99 budget a measurement with an unreliable p99 counts as within
   the budget only if that slow median is within it; the p99 of such a measurement sits on the
   cliff between the converged calls and the slow class and is not a stable number.
   **Budget membership** (both modes, since tuner 2.2 also with `--ler-target`).  A 2000-call p99
   is not a stable number where the tail is a ~1 % class (a host tail or the OSD path), so every
   screen survivor (a screen pass with `--ler-target`, a point above the LER floor without) whose
   latency-stage measurement is within 0.5x-2x of a budget (its largest measured/budget ratio) is
   re-timed before the in/out decision: at least 8000 calls per side and case (the latency
   confirmation's count) on its own fresh syndromes (seed stage `budget-retime`), paired with the
   latency stage's A/A reference in the ABBA schedule; that measurement decides, with the
   slow-class rule of a p99 budget.  At most 32 points are re-timed, nearest to the budget first;
   the rest of the margin keeps its latency-stage verdict and is listed (and warned about) as
   capped.  Outside the margin the latency-stage verdict stands.  `results.json` "budget_retime"
   records the margin, the reference, every re-timed point (latency-stage and re-timed p50/p99,
   the reference's p99, the verdict, the full measurement) and the wall time; the Pareto rows carry
   the decision's `budget_basis`; `pareto.md` has a section "Budget membership".
5. **Accuracy confirmation.**  The screen's passes are biased low (a point stops at its first
   favourable look), so the leading candidates are re-measured on fresh shots.  A separate seed
   stream, derived from `--seed` with its own stage name, samples a fixed number N of shots per
   candidate and case; N is declared before the first confirmation shot and recorded with its rule
   (the result of a doubling-then-bisection search for an N at which a point whose true LER is
   `target / 1.5` passes with probability at least 0.9; not necessarily the smallest such N,
   because the probability is not monotone in N; `--confirm-shots` overrides it, `--quick` caps
   it).  There is no early stop and every candidate
   decodes the same shots of a case.  The candidates are the screen survivors within the latency
   budget (after the re-timing), in the order the choice ranks them: the screen choice's point
   first, then the other in-budget points (fastest p50 first with `--ler-target`); a point over the
   budget is never confirmed.  They are confirmed one at a time -- the accuracy confirmation, then
   the latency confirmation -- until one is accepted or the in-budget list is exhausted
   (`--confirm-candidates K` caps the count, with a warning when the cap leaves candidates
   unconfirmed).  A candidate passes when the one-sided Clopper-Pearson upper bound at the tuner's
   confidence is below the target at every case; each candidate's latency confirmation (at least
   8000 fresh distinct syndromes per case in a full run) gives its `latency_within`.  With no `--ler-target` there is no bar: `passed` is true, a
   candidate is `accepted` when its latency confirmation is within the budget, and the 2048 fresh
   shots per case are a consistency check of the screen's estimate (`consistency`: the screen LER
   inside the confirmation's one-sided bounds at every case), not the reported LER.  The record is
   `results.json`
   "accuracy_confirmation": the planned shots and rule, the case seeds, and per candidate the
   screen's numbers, its per-case confirmation numbers, `passed`, `latency` (its latency
   confirmation: the pooled summary with `distinct_syndromes`, `syndrome_seeds`, the slow-class
   fields and `core_variant`, recorded for every candidate, chosen or not), `latency_within`,
   `accepted`, `failure_reasons` (why a candidate was not accepted: its accuracy bound against the
   target, its latency confirmation against each budget), `rank` and `resumed_case_shots` (the
   case-shots a `--resume` run reused for that candidate); and for the whole confirmation `chosen`,
   `in_budget_candidates` and `in_budget_order` (the candidate list), `confirmed` (how many were
   measured), `stopped` (`accepted`, `exhausted`, `cap` or `no candidate`), `unconfirmed` (the
   candidates a cap left out) and `max_candidates` (the cap, null without one).  A candidate whose latency confirmation has an unreliable p99 gets a warning, as does
   every Pareto row whose latency-stage measurement has one: `results.json` "warnings", the
   `pareto.md` section "Warnings" and the console.
6. **The Pareto table and the choice.**  `pareto.md` lists every measured point with its screen
   LER, interval, p50 and p99, and a section lists the confirmation per candidate.  The screen
   candidates are the points whose LER upper bound is below the target at every physical error
   rate and whose latency is within the budgets.  The fastest of them (lowest p50, then p99) sets
   the reference, but a difference smaller than the machine's own A/A band is noise, so the order
   is made by a **tie rule**: every candidate whose p50 is within the measured A/A band of the
   fastest (and, when the budget is a p99 budget, whose p99 is within the A/A band of the lowest
   p99) forms the tie set, and the winner inside it is the one with (1) the lower LER upper bound,
   then (2) fewer BP iterations, then (3) the alpha closest to the alpha of a shipped preset, then
   (4) the name order of the point label.  With `--weight`, the smallest score wins instead, and
   scores within `weight * log10(1 + p50 band)` of the best tie in the same way.  The screen's
   winner is the first candidate of the confirmation; the tie set in `pareto.md`, the console and
   `choice.tie_set` marks the accepted choice "<- chosen" (`chosen: true`), and the screen's pick
   "(screen pick)" (`screen_pick: true`, `choice.screen_choice_label`) when the confirmation passed
   it over; the Pareto table marks the rows CHOSEN and "screen pick (not accepted)".  The rule is
   deterministic given the measurements; the measurements themselves still vary from run to run.
   The **choice** is then the first confirmed candidate (in that order) that is `accepted`; its
   `ler`, `ler_upper`, `p50_us`, `p99_us`, `budget` and slow-class fields come from the
   confirmation and its latency confirmation, `ler_source` is `"accuracy_confirmation"`, and the
   screen's numbers are recorded next to it in `choice.screen`.  If no candidate is accepted there
   is no choice: the tool says so, lists the candidates and the nearest points (a confirmed
   candidate by its latency confirmation, after every point that did not fail its accuracy
   confirmation), writes `config.nearest.json` and no `config.json`, and exits with status 3.
   Each nearest entry's p50, p99 and budget ratio come from one measurement, which the entry names
   (`measurement`, with its call count): the latency confirmation of a confirmed candidate, the
   re-timing of a re-timed point, else the latency stage; a line reads "p99 P us, Rx the budget
   (re-timed at N calls per side)", with P / budget = R.
   **Without
   `--ler-target`** the candidates are the points above the LER floor and within the budget (after
   the re-timing), ranked by the weight score (or p50, then p99, without `--weight`); the choice is
   the first whose latency confirmation is within the budget, and the ones ranked above it are
   recorded in `choice.passed_over` with a warning.  Its LER is the screen's: `choice.ler`,
   `ler_lower`, `ler_upper` (`ler_interval`) and `ler_per_case` are the screen's fixed-size numbers
   (`ler_shots_per_case`, 131,072 by default) with their one-sided bounds, worst case over the
   cases, `ler_source` is `"screen"`, and `choice.confirmation_check` holds the confirmation's
   failures and its consistency verdict.  If no confirmed candidate is within the budget, or no
   point passes the floor and the budget, there is no choice: "no configuration qualifies", the
   nearest points (a confirmed one by its latency confirmation, the others by the decision's
   basis), `config.nearest.json`, no `config.json`, and exit status 3, exactly as with
   `--ler-target`.
7. **Outputs.**  Every number the tool prints is in `results.json`; the console output is a
   rendering of it.  The per-stage raw files keep the full chunk-by-chunk accuracy records, the
   latency runs and the accuracy confirmation.

Keep-warm is measured for the chosen configuration when the cadence is above 500 us: the same
handle is run with keep-warm off and on in an ABBA schedule, calling `Decoder.keep_warm()` every
500 us during the idle gap and stopping 50 us before each decode.  The decision rests on p50: it is
recommended when p50 improves beyond the A/A band (the clock effect shows up in the typical call).
The p99 ratio is recorded and must not exceed 1 plus its own A/A band, but this is **not a p99
guard**: at sparse cadences each side has only a few hundred calls, the p99 is one of the largest
three or four samples, and its A/A band is tens of percent, so only a large tail regression could
trip it.  The report says so next to the keep-warm line.  Below 500 us the call stream holds the
clock and the keep-warm tick does not fit before a call; the recommendation is then "off" with that
reason recorded.  Keep-warm is an application-side call, so it is recorded in `results.json`, not in
`config.json`.

**Core variants.**  On a GPU whose device class has two core variants (compute capability 8.9 with
at least 128 SMs: the package picks `ada64`, see docs/BUILD.md "Core variants") the chosen
configuration is then timed under both variants: a handle of each, created with
`ELDPC_DEVICE_CLASS` set around its create in the tuner's own process (the previous value is
restored), paired in the ABBA schedule on the same fresh syndromes, with an A/A of two handles of
the table's variant as the band.  `faster` is the variant whose p50 and p99 are both beyond the
band (worst case over the cases), otherwise "no difference"; when it is `default` the record's
`recommended_env` is `ELDPC_DEVICE_CLASS=default`, to set in the environment of the process that
creates the decoder.  The variant is not written into `config.json`: `Decoder(config=...)` has no
variant key, and the front reads the variable from the process environment at every create.  The
record is `results.json` "core_variants" (also in `pareto.md` and on the console); on a one-variant
device it says that the stage was skipped and why.

## Output files

| file | contents |
|---|---|
| `config.json` | the chosen ABI configuration, loadable with `eldpc.Decoder(config="config.json")` |
| `pareto.md` | the choice, the environment, and the table of every measured point |
| `results.json` | the full record: inputs, environment (the CUDA `toolkit` with its `source`: the loaded library's `build_info()`, or `nvcc --version` only when that names no toolkit; `nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv`, `CUDA_VISIBLE_DEVICES` and the devices `eldpc.devices()` reports), per-stage point records with chunk histories, seeds and `converged_share`, the `pruning` rule, latency experiments with their syndrome seeds, `distinct_syndromes` and slow-class fields, the `accuracy_confirmation` record, keep-warm, the latency `confirmation` of the final choice, the Pareto rows and the choice with its tie set; `budget_retime` (both modes, with a latency budget) and, without `--ler-target`, `ler_floor` |
| `raw/stage-coarse.json`, `raw/stage-refine.json` | the accuracy stages, one record per point, with every chunk's shots, failures and one-sided bounds |
| `raw/latency.json` | every paired latency experiment, its per-run blocks, ratios, syndromes and the A/A band |
| `raw/accuracy-confirmation.json` | the fixed-shot confirmation: planned shots, rule, case seeds and every candidate's per-case numbers |
| `raw/confirmation-checkpoint.json` | the confirmation progress saved after every decoded chunk; `--resume` continues from it, and it is removed when a run completes |
| `raw/confirmation.json`, `raw/keep-warm.json` | the latency confirmation of the chosen configuration and the keep-warm comparison, when they ran |
| `raw/environment.json` | the environment block, repeated in every results file |
| `raw/checkpoint.json` | accuracy progress, written after every evaluated group and removed when a run completes; `--resume` continues from it |
| `config.nearest.json` | only when nothing qualifies: the config of the nearest point (nearest to the latency budget when points pass accuracy but miss the budget -- a confirmed candidate by its latency confirmation, one that failed its accuracy confirmation last -- otherwise the point with the smallest LER upper bound).  It is recorded for reference and is not a qualifying configuration |

**Exit status** (both modes, with or without `--ler-target`): **0** -- a choice was made and
`config.json` written; **3** -- no configuration qualifies: `results.json`, `pareto.md` and
`config.nearest.json` (when there is a nearest point) are written, `config.json` is not; **any
other non-zero** -- an error or an abort (1: the run failed or the prompt was declined; 2: bad
arguments; 130: interrupted).  `eldpc-tune --help` repeats it.

Every results file names the command that produced it (`tool.command` in `results.json`, the
`command` field of the markdown) and carries the environment block (GPU, driver, compute capability,
the CUDA toolkit of the loaded library and where it was read, `CUDA_VISIBLE_DEVICES`, the devices `eldpc.devices()` reports, CPU, library and host
ISA, package version, Python/numpy/scipy/stim versions).

A worked example of these files is in `tuning/results/bb72-weight0/` (the weight-only command of
TUNER.md section 5; `config.json`, `pareto.md`, `results.json` and a `RUN.md`; the `raw/` files are
not committed, `results.json` holds every number).  It was written by tuner 2.0, so it lacks the
fields that 2.1 and 2.2 add (`ler_floor`, `budget_retime`, the environment's `toolkit`, the
confirmation counts and `failure_reasons`, the nearest entries' `measurement`).  The README's tuner
example (`--dem model.dem --ler-target 1e-3 --auto` on the code-to-DEM quick start's model) was run
with tuner 2.2 on an RTX 6000 Ada with a dual AMD EPYC 7542 host (a shared host; the latencies are
screens) at two budgets, `tuning/results/readme-example-rtx6000ada/budget-250us/` and
`budget-200us/` (the same files, `raw/` not committed).  The accuracy confirmation's shots are
seeded, so its counts are the same on every machine: (0.5, 11) and (0.5, 15) fail it (82 and 80
failures in 99,992 shots, upper bounds 0.00106 and 0.00103 against the target 0.001) and
(0.5, 31) passes it (76 failures, upper bound 0.000989).  With `--p99-budget-us 250` the run chose
(0.5, 31) (latency confirmation p50 / p99 183.6 / 217.7 us; exit status 0); with
`--p99-budget-us 200` no candidate was accepted, and the run named (0.5, 31) the nearest point
(p99 212.1 us, 1.06x the budget, re-timed on 8000 calls), wrote `config.nearest.json` and exited
with status 3.  Two small tools work on such a directory:
`tuning/tools/make_run_note.py --results DIR/results.json --commit C --label TEXT` writes the
`RUN.md` (the command and the environment from the results file, the commit only when `--commit`
gives it, and the tool's own name), and `tuning/tools/render_pareto.py DIR/results.json` rewrites
`pareto.md` from `results.json` with the current renderer (it computes nothing).

## Reading the outputs

- `LER` is the fraction of shots whose decoded observables differ from the sampled observables in
  at least one bit.  The interval next to it is the one-sided 99% Clopper-Pearson interval at the
  number of shots actually decoded; small shot counts give wide intervals, which is why a quick
  run proves only loose targets.  **The screen's LER and the confirmation's LER are different
  numbers**: with `--ler-target` the screen stops at a favourable look and is biased low, and the
  number to report is the confirmation's (`choice.ler`, `accuracy_confirmation.candidates[*].ler`);
  the screen's is recorded next to it and labelled as the screen.  Without `--ler-target` no point
  stops early, so the number to report is the screen's 131,072-shot estimate (`choice.ler`,
  `ler_source: "screen"`) and the confirmation's small sample is the consistency check.
- `p99_unreliable` marks a p99 that sits on the cliff between the converged calls and the slow
  class (`slow_fraction` in [0.005, 0.02]); the tool warns, and a p99 budget is then judged on the
  median latency of the slow calls (`slow_median_us`), recorded in the budget check.
- A point with zero observed failures still has a positive upper bound: the smallest bound provable
  with n shots and zero failures is about 4.6/n.  A full run derives its shot limit from a target
  below the default's floor (about 4.7/target); with `--quick` or an explicit `--max-shots` the
  limit is fixed, and the tool warns when the target is below its floor because then no point can
  qualify.
- `p50`/`p99` are empirical quantiles (numpy's linear interpolation) of the timed single-shot
  calls; the call count is recorded with every measurement.  The A/A band is the machine's own
  run-to-run spread over the same measurement schedule; latency differences smaller than the band
  are reported as "within_noise", not as a win.
- The Pareto star marks a point on the measured (LER, p50) frontier among measured points.

## Tests

```sh
python -m pytest tuning/tests
```

Run from the repository root.  Tests that need the built-in recipes skip when `qldpc` is not
installed, and tests that need a decoder skip when none can be created.

The interval tests check the beta quantiles against the defining binomial equations; the stopping
tests drive the sequential rule with scripted failures; the grid tests pin the fixed grid, the
refinement and the dominance rule; the choice tests cover the budget and weight selection and the tie rule (the p50 and p99 bands,
the order inside a tie set, and independence of the input order); the
determinism tests rerun an accuracy stage with the same seed, check that concurrent evaluation
(workers 1 versus 2) and a resumed partial stage produce identical records, and that dominance
survives grouping; the checkpoint tests cover the round trip, the refusal of a checkpoint with
another problem, grid, tuner version or stage point list, and the counts of a resumed stage; the
confirmation tests cover the shot-count rule (the N the search returns meets the margin and
N - 1 does not, a smaller N can also meet it, the override and the quick cap), fresh confirmation seeds, the fixed shot count with no early stop, verdicts
against hand-computed Clopper-Pearson bounds, determinism, the pruning rule on both sides of its
threshold and the slow-class budget rule; a run-schema test runs a quick tuner end to end and
checks the `eldpc.tune.v2` records, that the reported LER and the point come from the first
accepted candidate, the converged shares and the undecomposed DEM default; the round-trip test
writes and reloads `config.json` and compares decode outputs; the weight-mode tests reproduce the
four defects of tuner 2.0's `--weight 0` path on synthetic records, fake stages and decoder
doubles (an over-budget confirmation never becomes the choice, and the next confirmed candidate
does; BP alone below the `ms30` floor is never chosen or timed, and `--max-ler` replaces the
floor; the claim is the screen's estimate with the confirmation as a consistency check; a lucky
latency-stage p99 inside the margin is decided by an 8000-call re-timing paired with the
reference), check the exit status and the missing `config.json`, and run two quick GPU runs (a
choice above the floor; an impossible budget with no choice); the confirmation-loop tests reproduce
tuner 2.1's failures in both modes (the `--ler-target` path re-times the points near the budget
with the same margin, cap and warning, so a lucky 2000-call p99 no longer makes a candidate; the
confirmation continues past the third candidate and chooses the fifth, a `--confirm-candidates`
cap that leaves candidates unconfirmed warns and names them, an exhausted list gives exit status
3; a nearest line prints the p50 / p99 of the measurement behind its ratio and names it; the tie
set marks the accepted choice and the passed-over screen pick); the environment tests check that
the toolkit comes from the library's `build_info` and that `nvcc` is asked only without one; the latency tests cover the paired
syndrome sequence and the A/A cycle bookkeeping; the problem tests cover the recipes, the file
readers, the DEM options and the uniform-noise override; the report tests cover the writers and
stale-output removal.
