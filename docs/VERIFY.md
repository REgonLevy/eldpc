# Verifying eldpc

`verify/` is the public verification suite. It answers one question with a pass/fail table: are
`eldpc`'s corrections and observable predictions bit-identical to the reference decoder on corpora
that the user generates from public seeds on their own machine?

The suite is self-contained. It builds its own reference decoder from sources in `verify/reference`,
generates its corpora with stim and qLDPC, and uses `eldpc` only through the public Python API
(`eldpc.Decoder`, `eldpc.presets()`, `eldpc.load()`). It tests whatever `import eldpc` finds, so
install the package first (`pip install .`, see `BUILD.md`). It does not import any part of the
project the decoder was developed in.

## What is compared

For every problem, every preset configuration and every shot:

| comparison | pass criterion |
|---|---|
| `eldpc` single-shot correction vs the reference | must be identical |
| `eldpc` single-shot observables vs the reference | must be identical |
| `eldpc` `decode_batch` correction vs the reference | must be identical |
| `eldpc` `decode_batch` observables vs the reference | must be identical |
| `eldpc` single-shot vs `decode_batch` (correction, observables, status) | must be identical |
| `eldpc` status word vs a separately built library of the same decoder sources (only with `--reference-lib` or `ELDPC_REFERENCE_LIB`) | must be identical |
| the installed upstream `ldpc` 2.4.1 wheel vs the reference | informational only |

The status word is compared with the reference as well and reported, but the pass criterion for
status is the separately built library: the exactness contract requires bit-identical corrections
and observables against the reference and a bit-identical status word against a separate build of
the decoder sources at the same settings. Without that library the status words are compared with
the reference only; the run says so on the console, in its final line and in `summary.md`.
"Status words: the separately built library" below says how to build the library on your machine.

`run_verify.py` exits 0 if and only if every row has 0 mismatches on the required comparisons; it
exits 1 when any row differs, and 2 on a usage, environment or build error.

### Status words: the separately built library

Each decoder core of this repository is a decoder ABI v1 library: it exports the `dbench_*` entry
points the runner and `tests/test_bit_identity.py` speak. A core built separately with CMake and
loaded directly, without the package's C API front, is the separately built library. From the
package root:

```bash
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -S . -B build-ref -DCMAKE_BUILD_TYPE=Release
cmake --build build-ref -j --target eldpc_core_avx2      # build-ref/lib/libeldpc_core_avx2.so
export ELDPC_REFERENCE_LIB=$PWD/build-ref/lib/libeldpc_core_avx2.so
python verify/run_verify.py --quick                      # status words compared as well
python -m pytest tests/test_bit_identity.py             # the bit-identity tests run instead of skipping
```

The build needs the same toolchain as the package (docs/BUILD.md) and takes about a minute. Name
the package's toolkit in `CUDACXX` for the `cmake -S` step, as for the install (README, "CUDA 13
toolkit"): without it CMake takes the `nvcc` on `PATH` or `/usr/local/cuda`, which on a machine
with several toolkits can be another one than the package was built with, and the comparison would
then be with a build from a different toolkit. An existing `build-ref/` keeps the compiler it was
configured with; configure a new directory to change it. `run_verify.py` prints the `cuda=` field
of both libraries (`separately built library: cuda=NVIDIA 13.3.73; the package's GPU cores:
cuda=NVIDIA 13.3.73`) and says when they differ, and `summary.json` records both build-info
strings and the total of the status-word mismatches against the library
(`totals.status_mismatches_vs_reference_lib`; `null` without a library). The
AVX2 default core runs on every supported host, so it can serve as the reference for either host
core and for either core variant; on an Ada part with at least 128 SMs, where the package's handles
use the `ada64` core, the comparison also checks that variant against the default build.

What this comparison checks: the installed package -- its C API front, its host-ISA and core-variant
choice, its mapping of the decoder ABI and the status word, and the Python layer -- against a second
build of the same decoder sources made on your machine. What it does not check: the separate build
uses this repository's CMake rules, the same as the package's, so it does not test the build rules
themselves; the decoder's original reference build is not part of this repository. The runner
refuses the installed package's own libraries (`libeldpc.so` and the cores next to it) as the
separately built library, because they would compare the package with itself.

### The upstream wheel column

The comparison with the installed `ldpc` wheel is informational and never decides pass or fail. It
exists because upstream's OSD column sort leaves the order of equal-valued columns unspecified,
while the reference and `eldpc` use an explicit total order (ascending soft value, ties by
ascending column index, NaN after every number). A differing shot is classified by the same
tie-free rule the decoder was validated with:

- a shot that BP converged on (or a `decoder=bp` shot) never reaches post-processing, so its
  result cannot depend on the OSD order;
- otherwise the OSD column order is walked in groups of equal soft value. A group that contributes
  both pivot and non-pivot columns is where the tie order can change the pivot set; such a shot is
  labelled `tie_order`. A differing shot that is tie-free is labelled `unexplained`.

Every differing shot appears in `summary.json` with its cause; `summary.md` lists them in a table.
The number of differing shots is a property of the installed wheel and of the corpora, so the
summary of your run is the record; the column does not enter the pass criterion.

## The corpora

The suite's configuration (`verify/config.json`) is the single source of the problem list, the
shot counts per problem and mode, and the preset lists per mode. Its base seed is public and fixed
(`20261006`). The shipped file defines nine problems, all taken from the reference construction's
problem set. The sizes below are those of corpora built with qLDPC git main commit `4c4f801`
(0.4.2.dev20261004, the revision of the `qldpc` extra) and stim 1.16.0:

| problem | family | basis | rounds | m | n | k |
|---|---|---|---:|---:|---:|---:|
| `bb72_unif_p1e-3_r6_z` | bivariate bicycle l=6, m=6 | z | 6 | 252 | 2592 | 12 |
| `bb72_unif_p1e-3_r6_z_loss1e-2` | bivariate bicycle l=6, m=6, heralded erasure p=1e-2 | z | 6 | 324 | 2376 | 12 |
| `bb144_unif_p1e-3_r12_z` | bivariate bicycle l=12, m=6 | z | 12 | 936 | 10368 | 12 |
| `bb144_unif_p1e-3_r12_z_win3c1` | window (rounds 1-3) of the previous problem | z | 12 | 216 | 2952 | 12 |
| `bb144_unif_p1e-3_r12_xyz` | bivariate bicycle l=12, m=6 | joint X+Z | 12 | 1872 | 75600 | 24 |
| `bb288_unif_p1e-3_r18_z` | bivariate bicycle l=12, m=12 | z | 18 | 2736 | 31104 | 12 |
| `hgp98_unif_p1e-3_r4_z` | hypergraph product 7x7 | z | 4 | 245 | 2352 | 18 |
| `hgp98_unif_p1e-3_r4_xyz` | hypergraph product 7x7 | joint X+Z | 4 | 490 | 16954 | 36 |
| `sc9_si1000_p1e-3_z` | rotated surface code d=9, SI1000 p=1e-3 | z | 9 | 720 | 15954 | 1 |

All but the surface code use uniform depolarizing noise at p = 1e-3. The circuits of the other
eight problems are built by qLDPC, and other qLDPC revisions can build other circuits for the same
code and noise: with the released qLDPC 0.4.1, for example, the bb72 problem has n = 2664 instead of
2592 (the suite's own circuit builder run under qLDPC 0.4.1 and stim 1.16.0; bb144 xyz then has
n = 76032). The comparison is valid on any corpus, but only the tested revision reproduces the sizes
above; the run prints the qLDPC and stim it used, and `summary.json` records them.

Two problems exercise paths beyond the plain memory experiments. The loss problem inserts
`HERALDED_ERASE(p_erase)` on the final data qubits before the last measurement and one detector per
heralded qubit immediately after it (the reference construction's final-herald rule), then derives H,
O and the priors from the resulting DEM like every other problem. Because `HERALDED_ERASE` writes one
measurement record per qubit, every record target `rec[-k]` after the final measurement that points
past that measurement's own t records (k > t) is rewritten to `rec[-(k + t)]`, so the final detectors
still compare the final readout with the earlier round they were written for. The reference construction's
own insertion copies those targets unchanged, which makes some final detectors read the herald
records instead (with p_erase = 0, per-detector firing rates then differ from the clean circuit's by
up to 0.12 on bb72, against 0.0015 with the shift, over 200,000 shots;
`tools/results/heralded-loss/heralded-loss.json`); the suite applies the shift, and `tests/verify/test_herald.py` checks that
with p_erase = 0 the inserted circuit's DEM describes exactly the clean circuit's error distribution
(the same symptoms with the same probabilities once mechanisms of equal symptom are combined; stim
lists some of them merged in the inserted circuit) and that the herald detectors never fire. The window problem is derived
from its parent's stored corpus entry: the detectors of rounds 1 to 3 and the mechanisms whose
column touches them, with the parent's shots restricted to those detector rows; its DEM text is
rebuilt from the extracted arrays. Both rules match the reference construction's implementation, the loss rule after
the record shift is applied to the reference construction's circuit (checked by `tests/verify/test_dem_rule.py`
when the reference construction's Python package is available, see Tests below).

**What the loss problem verifies.** The loss row verifies the construction and the decoding of the
matrices the construction produces: the herald rows are part of H and the decoder receives static
priors. It does not verify loss-aware decoding. A loss workflow that sets per-shot priors from the
fired herald detectors and decodes only the non-herald rows is a different path; no row of this
suite exercises it, so a pass on the loss row must not be read as "loss decoding is verified".

The circuit constructions follow the reference construction: `stim.Circuit.generated` for the
surface-code memory experiment, qLDPC's `get_memory_experiment` with `BBCode`/`HGPCode` for the
others, the `DepolarizingNoiseModel` / `SI1000NoiseModel` noise models, and stim's
`detector_error_model(decompose_errors=..., approximate_disjoint_errors=True,
ignore_decomposition_failures=...)` with `decompose_errors` set for the surface code. H, O and the
priors are derived from the DEM by the reference construction's rule: one column per `error` instruction of
`dem.flattened()`, in order, never merged; the column's detector and observable entries are the
parities of the target occurrences; the prior is the instruction's probability.

Sampling is one seeded `compile_detector_sampler(seed=...).sample(shots=N,
separate_observables=True, bit_packed=True)` call per problem. The shot-set seed is
`derive_seed(base_seed, "dev", problem_id, "latency")` -- the reference construction's derivation -- so with the
same base seed, problem and shot count the suite generates exactly the shot set the reference construction
generates for that problem.

Shot counts are per problem and mode (`shots.quick` and `shots.full` in `config.json`), and each
mode has its own preset list (`configs.quick` and `configs.full`). In the shipped file `--quick`
uses 200 shots for each of the nine problems and the presets `a03125c3` and `ms30` (3,600 decodes
per library), and `--full` uses 500 shots for each problem and the presets `a03125c3`, `ms30`,
`a05c11`, `a0375c7` and `a05ms31` (22,500 decodes per library). The totals are always the sum over
the configured problems times the number of presets; `summary.md` of a run states the exact figures.

Generated corpora are cached under the cache directory (`$ELDPC_VERIFY_CACHE`, else
`$XDG_CACHE_HOME/eldpc/verify`, else `~/.cache/eldpc/verify`), keyed by the problem object, base
seed and shot count and by the generator: the qLDPC version (with the git commit it was installed
from, when pip recorded one), the stim version and stim's SIMD backend. Installing another qLDPC or
stim therefore builds new entries and never reuses a corpus that another version generated. An entry
is reused only when its manifest names the same generator and every file matches the SHA-256 in its
manifest; delete the cache directory to force regeneration. The run prints the generator and, for
each corpus entry, whether it was built or reused.

Corpus bytes are reproducible on one stim build and SIMD backend; stim does not promise identical
seeded samples across versions or backends (`stim_simd` is recorded in each corpus manifest). On
another stim build the corpus differs, and the comparison with the reference remains valid on that
corpus, but byte-level corpus digests are not portable.

## The reference decoder

The reference is ldpc 2.4.1's BP+OSD-0 in the deterministic form the decoder was developed
against: the C++ headers of ldpc 2.4.1 (upstream commit
`d3429964cd4ffe1abfc041c6ec8b8425cb174f40`), with the reference construction's tie-order patch
applied, behind the same C ABI shim the reference construction's reference library uses. The patch gives the OSD
column sort an explicit total order (ascending soft value, ties by ascending column index, NaN
after every number) and fixes the LSD candidate order and shuffle seed; BP itself is untouched. The
patch file, the MIT licence and a NOTICE are in `verify/reference/`.

`run_verify.py` builds the reference on first use into the cache directory, keyed by the SHA-256 of
every reference source file, and reuses it while those files are unchanged. It needs CMake >= 3.16
and a C++17 compiler; set `ELDPC_VERIFY_CMAKE` if CMake is not on `PATH`. The build uses
`-O2 -DNDEBUG -ffp-contract=off -fno-fast-math`, the same determinism flags as the reference construction
reference. The library's `dbench_decoder_build_info()` string -- compiler, flags, build type,
`ldpc=2.4.1` and the patch SHA-256 -- is recorded in `summary.md` and `summary.json`.

The packaged reference can be compared with the reference construction's own build of the reference
decoder on the same shots by `tests/verify/test_reference.py` when `ELDPC_REFERENCE_LDPC_LIB`
names that library (a `libdbench_ldpcref.so`). That needs a build of the reference construction, so it is optional and
not part of the public suite; the suite compares against the packaged reference only.

## Running it

Requirements:

- a CUDA GPU supported by the installed `eldpc` (the README, "Supported GPUs"), or `--device cpu`
  to compare the CPU core's outputs instead (no GPU needed; the CPU core must return the same
  outputs, so the comparison and its PASS line mean the same for it);
- the package and its dependencies (numpy, scipy, stim), plus qLDPC for the corpora: from the
  package root, `pip install ".[qldpc]"` installs the tested qLDPC revision and `ldpc` 2.4.1 (see
  the README, "Install"); without qLDPC the runner exits 2 with that command;
- CMake >= 3.16 and a C++17 compiler for the reference build (CPU only);
- optionally, the `ldpc` 2.4.1 wheel for the informational column.

From a checkout of the repository, with the package installed:

```bash
python verify/run_verify.py --quick
```

The suite lives in the source tree and is not part of the wheel; the script only needs
`import eldpc` to reach the installed library. To compare status words with a separately built
library of the same decoder sources (built as in "Status words: the separately built library"
above), either pass it or set the variable:

```bash
ELDPC_REFERENCE_LIB=build-ref/lib/libeldpc_core_avx2.so python verify/run_verify.py --quick
python verify/run_verify.py --quick --reference-lib build-ref/lib/libeldpc_core_avx2.so
```

Options:

```
--quick | --full        suite size (default: --quick); contents defined by the suite config
--configs A,B           subset of preset names (default: the mode's list)
--problems A,B          subset of problem ids (default: every configured problem)
--out DIR               summary.json and summary.md go here (default: eldpc-verify-out)
--device gpu|cpu|auto   the engine whose outputs are compared (default: gpu, the GPU core); cpu runs
                        the CPU core (a machine without a supported GPU), auto the GPU when it is
                        usable and else the CPU core; every row records the engine it ran on
                        ("device"), and the PASS/FAIL line names a non-GPU engine
--lib PATH              passed to eldpc.load(); the package's C library to exercise
--reference-lib PATH    a separately built library of the same decoder sources (decoder ABI v1);
                        enables the status comparison (default: $ELDPC_REFERENCE_LIB; an empty
                        value disables it)
--config PATH           suite configuration (default: verify/config.json)
--cache DIR             corpus and reference cache (default: see above)
--jobs N                parallel jobs for the reference build
--no-upstream           skip the informational upstream-wheel comparison
```

The run first prints the corpus generator (the qLDPC and stim versions) and whether a separately
built library was given, then one line per problem and preset as it goes, and ends with `PASS` or
`FAIL` and the comparisons that were made. A quick run without a separately built library ends
with a line of this form:

```
PASS: 18 rows, 3600 shots, 0 mismatches; corrections and observables vs the reference; status words not compared (no separately built library, see docs/VERIFY.md) (<seconds> s)
```

With one, the line says `corrections and observables vs the reference, status words vs the
separately built library`.

## Reading the output

`DIR/summary.json` contains:

- `status`: `pass` or `fail`;
- `rows`: one object per (problem, preset) with `shots`, `mismatches`, the per-stream counts
  (`correction_mismatches`, `observables_mismatches`, `batch_correction_mismatches`,
  `batch_observables_mismatches`, `single_batch_mismatches`, `status_mismatches_vs_reference`,
  `status_mismatches_vs_reference_lib`), the first mismatched shot indices, per-phase timings, and the
  `upstream` object (`differing_shots`, `correction_mismatches`, `observables_mismatches`,
  `causes`, and the list of differing shots);
- `totals`, `timings`, `selection` (mode, config path, seed, problems, presets, shot counts);
- `environment`: GPU name/driver/compute capability, CPU, Python and package versions, `eldpc`
  version/build info/ISA/devices, the reference build description and source hash, and the
  description of the separately built library when one was given.

`DIR/summary.md` is the human-readable version: the environment header, the pass/fail result, the
row table, the status-word table, the list of upstream-wheel differences with their causes, and
the timing breakdown. It is the record of a run; this repository does not ship a results file.

## Timing

The reference decodes on the CPU and dominates the wall time of a run; `eldpc` decodes the same
shots in a small fraction of it. The joint X+Z bb144 problem is the slowest row. A second run with
a warm cache skips the reference build and the corpus generation, but it is only somewhat faster
(232 s and then 201 s for `--quick` on an L40S with a dual Xeon Silver host), because the CPU
reference decoding dominates. A first `--quick` run takes about 4 minutes there and prints nothing
for several minutes while it builds the reference and the corpora. Wall times depend on the machine
and its load, and the per-phase times are in the `timings` section of `summary.json`.

## Tests

`tests/verify` holds tests of the suite's own machinery. They need CMake and a C++ compiler (they
build the reference) but no GPU; the ones that build corpora need qLDPC and skip without it:

```bash
python -m pytest -q -rs tests/verify
```

They check that the corpus generator is deterministic (two builds of the same entry are
byte-identical, a cache hit returns the same arrays, different shot counts are separate entries),
that the packaged reference builds from its own directory, reuses its cache, decodes correctly
(zero-syndrome short circuit, syndromes in the column space of H, observables equal to
O*correction, batch equal to a loop, deterministic repeats, invalid configs rejected), and that the
row comparison counts a deliberately corrupted correction.

Two optional comparisons depend on artifacts outside the repository, so the number of passed and
skipped tests depends on them; `-rs` prints the reason of every skip:

| variable | enables | when unset |
|---|---|---|
| `ELDPC_REFERENCE_PYTHON` | `test_dem_rule.py`: the suite's circuits, DEMs, matrices, priors and seeded shots against the reference construction's corpus code (point it at the reference construction's `python/` directory) | the whole module is skipped |
| `ELDPC_REFERENCE_LDPC_LIB` | the last test of `test_reference.py`: the packaged reference against the reference construction's own reference build | that one test is skipped |

A run with both unset shows skips for exactly these tests and no failures; that is a normal result,
not a partial one, but it does not include the reference construction comparisons.

## Optional tools

`verify/tools` holds scripts that reuse the suite's corpus builder and reference; they need the
installed `eldpc` package and are not part of the pass/fail suite:

- `soft_output_check.py` compares `eldpc`'s posterior LLRs with the reference's soft output
  (needs a GPU);
- `upstream_divergence_check.py` runs the upstream-wheel comparison on larger public-seeded shot
  sets and reports any differing shot with its cause (needs the `ldpc` wheel);
- `tie_classification_cost.py` measures the per-shot cost of the tie-free classification that
  explains a difference (needs the `ldpc` wheel).

Run them from the repository root, for example
`python verify/tools/soft_output_check.py --configs a03125c3,ms30 --out soft.json`.
