# The README's tuner example on an RTX 6000 Ada with a dual EPYC 7542 host

Two runs of tuner 2.2 (revision `480f488`) on the model that the README's quick start "From a code
to a circuit, a DEM and a decode" writes (`model.dem`, 153,098 bytes, sha256
`f88bcdab4aa3ffd2954e9a4484c06c4bf763ab45901d13d3a6cf4a64b095b0b6`, with qLDPC `4c4f801` and stim
1.16.0), with the README's command at two latency budgets:

| directory | command | exit status | outcome |
|---|---|---:|---|
| `budget-250us/` | `python tuning/eldpc_tune.py --dem model.dem --ler-target 1e-3 --p99-budget-us 250 --auto --out tune250` | 0 | chose (0.5, 31): 76 / 99,992 confirmation failures (upper bound 0.000989); latency confirmation p50 / p99 183.6 / 217.7 us; 211 s |
| `budget-200us/` | the same with `--p99-budget-us 200 --out tune200` | 3 | "No configuration qualifies"; nearest (0.5, 31) at p99 212.1 us, 1.06x the budget (re-timed on 8000 calls); `config.nearest.json`; 182 s |

In both runs (0.5, 11) and (0.5, 15) fail the accuracy confirmation (82 and 80 failures in 99,992
shots, upper bounds 0.00106 and 0.00103 against the 1e-3 target); those shots are seeded, so the
counts do not depend on the machine. The latencies are screens of a shared host (`RUN.md`). Each
directory holds the run's `results.json` (the GPU's UUID removed), `pareto.md`, the config file it
wrote and a `RUN.md` from `tuning/tools/make_run_note.py`; the `raw/` checkpoints are not
committed.
