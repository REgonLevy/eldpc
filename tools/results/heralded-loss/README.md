# Heralded-loss problems: construction and accuracy

`heralded-loss.json` is the summary of a study of final-measurement heralded erasure (`HERALDED_ERASE` before the
last measurement, one herald detector per qubit):

- **the construction**: `HERALDED_ERASE` on t qubits writes t measurement records, so every later `DETECTOR` or
  `OBSERVABLE_INCLUDE` target `rec[-k]` with k > t must become `rec[-(k+t)]`. The corrected insertion applies that
  shift (as `verify/eldpc_verify/circuits.py` does); the uncorrected one copies the targets unchanged, so some final
  detectors read herald records. For each of the six loss problems the file records the rewrite audit (instructions,
  targets rewritten), the check at p_erase = 0 (the corrected circuit's DEM equals the clean circuit's flattened DEM
  coefficient for coefficient, the uncorrected one does not; the largest per-detector firing-rate difference to the
  clean circuit over 200,000 shots per circuit, against a Bonferroni-union Hoeffding bound) and the problem sizes of
  both constructions;
- **accuracy on the six loss problems** with both constructions: the eldpc core at `a03125c3` and `a05ms31` and the
  ldpc 2.4.1 reference at `ms30` (stock settings), all decoders of a problem on the same shots;
- **herald use on a d = 3 surface code** (3 rounds, circuit noise p = 1e-3, final heralded erasure 0.1): ldpc
  BP+OSD-0 at the `a03125c3` settings in four modes (blind; herald detectors in the DEM with static priors, which is
  what the package decodes; per-shot priors from the fired heralds on the non-herald rows; per-shot priors on all
  rows), 100,000 shots per mode, the same shots for the four modes, with exact McNemar tests between the modes and the
  comparison with an earlier 4,000-shot run.

## How it was produced

- Machine E of `tuning/KNOWLEDGE.md` section 0.2 (RTX 5080, driver 595.91.07, Ryzen 9 7950X3D, CUDA 13.3,
  stim 1.16.0); accuracy only.
- Decoders: the measurement-harness build of the frozen `fastbpgpu6g` core through its batch decode call, and the
  harness's ldpc 2.4.1 reference library (checked against the `ldpc` 2.4.1 Python package on 5,000 shots: identical
  corrections and observables).
- Stopping rule, fixed before the first shot and checked at whole 100,000-shot chunks: at least 100,000 shots, then
  stop at 100 failures, or at 2,000,000 shots. The uncorrected problems were decoded at exactly 100,000 shots per
  decoder. Intervals are 95 % Clopper-Pearson; the decoders are not compared shot by shot (no paired test) on the
  loss problems.
- The file holds counts, rates, intervals and test statistics only; shot sets and corpora are not included.

## Cited by

- `tuning/KNOWLEDGE.md` section 5.6 (the construction, the uncorrected bb72 LERs, the LER table, the readings, the
  herald-use table and its McNemar tests) and section 8 (known gaps).
- `docs/VERIFY.md`, "Problems" (the loss problem's construction and what the uncorrected insertion does at
  p_erase = 0).
- `docs/SUPPORT.md`, "No loss-aware workflow", and `README.md` ("It is not ... a loss-aware decoder"): the per-shot
  prior workflow measured here is not shipped.
