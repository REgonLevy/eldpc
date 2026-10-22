# Joint against separate X/Z decoding (accuracy only)

`joint-vs-separate.json` is the summary of a study that decoded the same shots of a circuit with both X and Z
detectors in two ways:

- **joint**: one handle on the circuit's full detector error model, one column per `error` instruction of
  `dem.flattened()` (what `Decoder.from_dem` builds);
- **separate**: the joint model projected onto the X-type and onto the Z-type detector rows, the columns with no
  detector in a sector dropped (none of them flips a sector observable), identical columns merged exactly
  (independent mechanisms combined by parity), each sector decoded by its own handle; the shot's prediction is the
  X-sector decoder's 12 X-type observables followed by the Z-sector decoder's 12 Z-type observables.

A shot fails when any of its 24 observables is wrong; the LER is per shot of the 12-round experiment. No timing was
measured (the separate route makes two decode calls per shot).

## How it was produced

- Machine E of `tuning/KNOWLEDGE.md` section 0.2 (RTX 5080, driver 595.91.07, Ryzen 9 7950X3D, CUDA 13.3.73,
  stim 1.16.0). LER does not depend on the machine: the decoder is bit-exact.
- Decoder: the measurement-harness build of the frozen `fastbpgpu6g` core (the same sources as `src/core/`),
  BP+OSD-0 with min-sum BP and the parallel schedule; the BP-only rows use `decoder=bp`.
- Problems: `bb144_unif_p1e-3_r12_xyz` (main), the same code at p = 3e-3, and `hgp98_unif_p1e-3_r4_xyz`, built
  with the development corpus's circuits (76,032 joint columns for bb144 xyz; the package's qLDPC revision builds
  75,600, and both merge to the same 71,280 columns).
- Main runs: 1,000,000-shot segments with a stopping rule fixed before the first shot (stop at a checkpoint when the
  two 99 % Clopper-Pearson intervals are disjoint): `a05ms31` stopped at 2,000,000 shots, `a03125c3` ran to
  5,000,000. Every other row is a fixed-size run on its own fresh shots: an independent confirmation per preset
  (1,000,000 shots), the crossed configs (alpha 0.3125, cap 31) and (alpha 0.5, cap 3) in two samples each
  (1,000,000 + 2,000,000 shots, also pooled), the other problems (1,000,000 or 200,000 shots), BP alone (200,000),
  and merged against unmerged sector projections (500,000).
- Paired test: exact two-sided McNemar (binomial) on the discordant shots; `verdict` is the decoding with the lower LER
  when p < 0.05, else `tie`. `separate_over_joint` is the separate LER divided by the joint LER (computed).
- The file holds counts, rates, intervals and test statistics only; shot sets are not included.

## Fields

`construction` (detector types, column and nonzero counts of the joint and the merged sector models), `rows` (the
BP+OSD-0 comparisons), `bp_only_rows`, `unmerged_projection` (the sector projections without the merge),
`joint_merged_control` (the joint model with its 4,752 duplicate columns merged), `per_sector_failures`,
`bp_convergence_main_run`, `main_run_checkpoints`, `stopping_rule`.

## Cited by

- `tuning/KNOWLEDGE.md` section 3.5 (the table, "Other rows of the same record" and the readings), section 0.4
  (workflow step 1) and section 8 (known gaps).
